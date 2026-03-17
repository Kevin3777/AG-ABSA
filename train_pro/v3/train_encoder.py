import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel,AutoTokenizer,get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import GradScaler,autocast
import json
import os
from tqdm import tqdm

# ==========================================
# 1. 模型架构创新：带复数投影层的 AnglE
# ==========================================
class AnglE_Pro(nn.Module):
    def __init__(self, model_name, pooling='cls'):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        
        # 获取 hidden_size (通常是 768)
        hidden_size = self.backbone.config.hidden_size
        
        # 【创新点 1】复数投影层 (Complex Projection)
        # 不直接使用 BERT 输出，而是先通过一个全连接层进行语义空间的转换
        # 这允许模型学习如何更好地将语义映射到复数平面
        self.complex_adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh() # Tanh 将数值限制在 [-1, 1]，增加复数计算的数值稳定性
        )
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        last_hidden = outputs.last_hidden_state
        
        # Pooling
        if self.pooling == 'cls':
            embedding = last_hidden[:, 0]
        else:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            embedding = torch.sum(last_hidden * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # 经过投影层，生成更适合复数运算的特征
        projected_embedding = self.complex_adapter(embedding)
        
        # 归一化 (对于 Cosine 计算很重要，对 Angle 计算也有助于稳定)
        return F.normalize(projected_embedding, p=2, dim=1)

# ==========================================
# 2. Loss 创新：Smart AnglE Loss (动态权重)
# ==========================================
class SmartAnglELoss(nn.Module):
    def __init__(self, tau=0.05, angle_w=1.0):
        super().__init__()
        self.tau = tau
        self.angle_w = angle_w

    def forward(self, query, pos, neg):
        """
        query, pos, neg: [Batch, Hidden]
        """
        # --- 准备工作：切分实部和虚部 ---
        half_dim = query.shape[1] // 2
        
        # 辅助函数：转复数
        def to_complex(tensor):
            return torch.complex(tensor[:, :half_dim], tensor[:, half_dim:])

        z_q = to_complex(query)
        z_p = to_complex(pos)
        z_n = to_complex(neg)

        # --- A. 计算 Cosine (用于基础对比 + 动态权重计算) ---
        # Cosine Similarity = Real(z1 * conj(z2)) / (|z1|*|z2|)
        # 在这里因为 inputs 已经归一化了，所以直接点积即可
        cos_sim_pos = torch.sum(query * pos, dim=1)
        cos_sim_neg = torch.sum(query * neg, dim=1)

        # --- B. 【创新点 2】动态权重 (Smart Scale) ---
        # 逻辑：如果 Cosine 相似度很高（接近1或-1），说明处于梯度饱和区
        # 我们就根据 Cosine 的饱和程度，动态放大 Angle Loss 的权重
        with torch.no_grad():
            # 饱和度：越接近 1，saturation 越大
            saturation = torch.abs(cos_sim_pos) 
            # 动态系数：从 1.0 到 2.0 之间变化
            dynamic_scale = 1.0 + (saturation ** 2)

        # --- C. 计算 Angle Difference ---
        # 论文 Eq 6: angle_diff = |angle(z / w)|
        # z / w = z * conj(w) / |w|^2
        
        # Query vs Positive
        # 添加 1e-9 防止除零
        diff_pos = torch.abs(torch.angle(z_q / (z_p + 1e-9)))
        
        # Query vs Negative
        diff_neg = torch.abs(torch.angle(z_q / (z_n + 1e-9)))

        # --- D. 组合 Loss ---
        # 1. Cosine Contrastive Part (基础能力)
        loss_cos = torch.log(1 + torch.exp((cos_sim_neg - cos_sim_pos) / self.tau)).mean()
        
        # 2. Angle Part (高精细度区分)
        # 注意：这里乘以了 dynamic_scale
        # 并且只针对 Positive Pair 做动态加权（我们希望正样本的角度对齐得非常准）
        angle_gap = (diff_pos - diff_neg) / self.tau
        loss_angle_raw = torch.log(1 + torch.exp(angle_gap))
        
        dynamic_scale = dynamic_scale.unsqueeze(1)
        
        # 应用动态权重
        loss_angle = (loss_angle_raw * dynamic_scale).mean()

        # --- E. 模长正则化 (新增) ---
        # 我们不赋予模长特殊含义，但希望它保持在 1 附近，保证数值稳定
        # 避免某些向量模长变成 1000 或 0.0001
        
        # 计算模长 (L2 Norm)
        mag_q = torch.norm(query, p=2, dim=1)
        mag_p = torch.norm(pos, p=2, dim=1)
        mag_n = torch.norm(neg, p=2, dim=1)
        
        # 让模长尽量接近 1
        loss_mag = torch.mean((mag_q - 1)**2 + (mag_p - 1)**2 + (mag_n - 1)**2)

        # 给一个很小的权重，比如 0.01
        return loss_cos + self.angle_w * loss_angle + 0.01 * loss_mag

# ==========================================
# 3. 数据处理 (保持不变，利用 Token Type IDs)
# ==========================================
class SentimentTripletDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        self.data = []
        # 健壮的文件读取
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            self.data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        else:
            print(f"Warning: File {data_path} not found.")
            
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text_to_input(self, text_obj):
        # 兼容两种输入：
        # 1. 你的新格式 JSON: {"query_text": "...", "query_aspect": "..."}
        # 2. 你的旧格式 String: "Aspect: ..., Context: ..."
        
        aspect = ""
        context = ""
        
        if isinstance(text_obj, dict):
            # 新格式
            aspect = text_obj.get('aspect', '')
            context = text_obj.get('text', '')
        elif isinstance(text_obj, str):
            # 旧格式解析
            try:
                parts = text_obj.split("Context:")
                aspect = parts[0].replace("Aspect:", "").strip().replace(",", "")
                context = parts[1].strip()
            except:
                context = text_obj
        
        # 【创新点 3】利用 Token Type IDs
        # inputs: [CLS] Aspect [SEP] Context [SEP]
        inputs = self.tokenizer(
            aspect, 
            context, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
            
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'token_type_ids': inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0)
        }

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 适配你的数据 Key
        # 如果 JSON key 是 'query_text'/'query_aspect' 等，这里要做个小映射
        # 假设数据已经是 {"query": "Aspect:..", "positive": "...", "negative": "..."}
        
        return {
            'query': self.text_to_input(item.get('query', item.get('query_text', ''))),
            'positive': self.text_to_input(item.get('positive', item.get('pos_text', ''))),
            'negative': self.text_to_input(item.get('negative', item.get('neg_text', '')))
        }

# ==========================================
# 4. 适配 Config 的训练主程序
# ==========================================
class AnglETrainer:
    def __init__(self, config_path):
        # 1. 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = json.load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Initializing AnglE Pro Trainer on {self.device}")
        
        # 2. 初始化模型与组件
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg['model']['name'])
        self.model = AnglE_Pro(
            self.cfg['model']['name'], 
            pooling=self.cfg['model']['pooling_strategy']
        ).to(self.device)
        
    def train(self):
        # 数据集
        data_path = self.cfg['data']['input_jsonl_file']
        dataset = SentimentTripletDataset(
            data_path, 
            self.tokenizer, 
            max_len=self.cfg['model']['max_length']
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.cfg['training']['batch_size'], 
            shuffle=True,
            num_workers=0, # Windows下建议设为0，Linux可设为4
            pin_memory=True
        )
        
        # 优化器
        optimizer = AdamW(self.model.parameters(), lr=self.cfg['training']['learning_rate'])
        
        # 学习率调度
        total_steps = len(dataloader) // self.cfg['training']['gradient_accumulation_steps'] * self.cfg['training']['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.cfg['training']['warmup_steps'], 
            num_training_steps=total_steps
        )
        
        # Loss (使用 Config 中的参数)
        # 注意：这里我们统一使用 tau=0.05
        loss_tau = self.cfg['loss'].get('tau', 0.05)
        loss_angle_w = self.cfg['loss'].get('angle_w', 1.0)
        
        loss_fn = SmartAnglELoss(tau=loss_tau, angle_w=loss_angle_w)
        
        # 混合精度训练 (FP16)
        use_fp16 = self.cfg['training'].get('fp16', False)
        scaler = GradScaler() if use_fp16 else None
        if use_fp16:
            print("⚡ Mixed Precision Training (FP16) Enabled")

        # 训练循环
        self.model.train()
        global_step = 0
        
        accum_steps = self.cfg['training']['gradient_accumulation_steps']
        
        for epoch in range(self.cfg['training']['num_epochs']):
            loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.cfg['training']['num_epochs']}")
            total_loss = 0
            
            optimizer.zero_grad()
            
            for step, batch in enumerate(loop):
                # 数据上 GPU
                q_in = {k: v.to(self.device) for k, v in batch['query'].items()}
                p_in = {k: v.to(self.device) for k, v in batch['positive'].items()}
                n_in = {k: v.to(self.device) for k, v in batch['negative'].items()}
                
                # --- 前向传播 (支持 FP16) ---
                if use_fp16:
                    with autocast():
                        q_emb = self.model(**q_in)
                        p_emb = self.model(**p_in)
                        n_emb = self.model(**n_in)
                        loss = loss_fn(q_emb, p_emb, n_emb)
                        loss = loss / accum_steps # 梯度累积归一化
                else:
                    q_emb = self.model(**q_in)
                    p_emb = self.model(**p_in)
                    n_emb = self.model(**n_in)
                    loss = loss_fn(q_emb, p_emb, n_emb)
                    loss = loss / accum_steps

                # --- 反向传播 ---
                if use_fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # --- 梯度更新 (累积步数到达时) ---
                if (step + 1) % accum_steps == 0:
                    if use_fp16:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                current_loss = loss.item() * accum_steps # 还原用于显示的 Loss
                total_loss += current_loss
                
                # Logging
                if global_step % self.cfg['training']['logging_steps'] == 0 and (step + 1) % accum_steps == 0:
                     loop.set_postfix(loss=f"{current_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                else:
                     loop.set_postfix(loss=f"{current_loss:.4f}")

            # Epoch 结束保存
            save_dir = os.path.join(self.cfg['data']['output_dir'], f"checkpoint-epoch-{epoch+1}")
            self.save_model(save_dir)

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
        # 保存 Backbone
        self.model.backbone.save_pretrained(path)
        # 保存 Tokenizer
        self.tokenizer.save_pretrained(path)
        # 保存 Pro 版特有的 Projection Layer
        torch.save(self.model.complex_adapter.state_dict(), os.path.join(path, "complex_adapter.pt"))
        # 保存 Config 副本
        with open(os.path.join(path, "train_config.json"), 'w') as f:
            json.dump(self.cfg, f, indent=2)
            
        print(f"✅ Checkpoint saved to {path}")

if __name__ == "__main__":
    # 确保有一个名为 train_config.json 的文件在同目录下，或者修改路径
    config_file = r"train_pro\v3\config\train_config.json"
    
    # 如果文件不存在，先创建一个示例（方便调试）
    if not os.path.exists(config_file):
        print("Config file not found, please check path.")
    else:
        trainer = AnglETrainer(config_file)
        trainer.train()