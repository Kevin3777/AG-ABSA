import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import json
import os
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler 
import gc

# ==========================================
# 1. 架构创新：残差复数适配器 (Residual Adapter)
# ==========================================
class AnglE_Res_Pro(nn.Module):
    def __init__(self, model_name, pooling='cls'):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        
        hidden_size = self.backbone.config.hidden_size
        
        # 【创新点 1】残差投影层
        # 不直接替换特征，而是学习一个“修正项”
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4), # 降维，减少参数量
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size), # 升维回原尺寸
            nn.Tanh() # 限制幅度，保证数值稳定
        )
        
        # 这是一个可学习的门控系数，初始化为 0
        # 这意味着：训练开始时，Output ≈ BERT Output。随着训练，Adapter 逐渐介入。
        self.gate = nn.Parameter(torch.zeros(1)) 

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
        
        # 【关键】残差连接： 原特征 + (门控 * 修正特征)
        # 初始时 gate=0，等效于标准 BERT，保证不崩
        adjustment = self.adapter(embedding)
        projected_embedding = embedding + self.gate * adjustment
        
        return F.normalize(projected_embedding, p=2, dim=1)

# ==========================================
# 2. Loss 创新：饱和感知动态权重 (Smart Scale)
# ==========================================
class SmartAnglELoss(nn.Module):
    def __init__(self, tau=0.05, angle_w=1.0):
        super().__init__()
        self.tau = tau
        self.angle_w = angle_w

    def forward(self, query, pos, neg):
        # 1. 转复数
        half_dim = query.shape[1] // 2
        def to_complex(tensor):
            return torch.complex(tensor[:, :half_dim], tensor[:, half_dim:])
        
        z_q = to_complex(query)
        z_p = to_complex(pos)
        z_n = to_complex(neg)

        # 2. 计算 Cosine (用于基础对比 + 动态权重检测)
        cos_sim_pos = torch.sum(query * pos, dim=1)
        cos_sim_neg = torch.sum(query * neg, dim=1)

        # 3. 【创新点 2】Dynamic Scale
        # 仅当 Cosine 很大(>0.9) 但模型还需要优化时，放大梯度
        with torch.no_grad():
            saturation = torch.abs(cos_sim_pos)
            # 限制放大倍数在 [1.0, 1.5] 之间，防止梯度爆炸
            dynamic_scale = 1.0 + 0.5 * (saturation ** 2)
            dynamic_scale = dynamic_scale.unsqueeze(1) # [Batch, 1]

        # 4. Angle Loss Calculation
        # 添加 eps 防止 NaN
        diff_pos = torch.abs(torch.angle(z_q / (z_p + 1e-9)))
        diff_neg = torch.abs(torch.angle(z_q / (z_n + 1e-9)))
        
        angle_gap = (diff_pos - diff_neg) / self.tau
        
        # LogSumExp 形式更加数值稳定
        loss_angle_raw = torch.log(1 + torch.exp(angle_gap))
        
        # 应用动态权重
        loss_angle = (loss_angle_raw * dynamic_scale).mean()

        # 5. Cosine Loss
        loss_cos = torch.log(1 + torch.exp((cos_sim_neg - cos_sim_pos) / self.tau)).mean()

        return loss_cos + self.angle_w * loss_angle

# ==========================================
# 3. 数据处理 (保持不变)
# ==========================================
class SentimentTripletDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        self.data = []
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try: self.data.append(json.loads(line))
                        except: continue
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def text_to_input(self, text_obj):
        aspect = ""
        context = ""
        if isinstance(text_obj, dict):
            aspect = text_obj.get('aspect', '')
            context = text_obj.get('text', '')
        elif isinstance(text_obj, str):
            try:
                parts = text_obj.split("Context:")
                aspect = parts[0].replace("Aspect:", "").strip().replace(",", "")
                context = parts[1].strip()
            except: context = text_obj
        
        inputs = self.tokenizer(
            aspect, context, 
            max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'token_type_ids': inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0)
        }

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'query': self.text_to_input(item.get('query', item.get('query_text', ''))),
            'positive': self.text_to_input(item.get('positive', item.get('pos_text', ''))),
            'negative': self.text_to_input(item.get('negative', item.get('neg_text', '')))
        }

# ==========================================
# 4. 训练流程 (v5)
# ==========================================
class AnglETrainer:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = json.load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Initializing AnglE v5 (Residual Pro) on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg['model']['name'])
        # 使用 Residual Pro 模型
        self.model = AnglE_Res_Pro(
            self.cfg['model']['name'], 
            pooling=self.cfg['model']['pooling_strategy']
        ).to(self.device)
        
    def train(self):
        dataset = SentimentTripletDataset(
            self.cfg['data']['input_jsonl_file'], 
            self.tokenizer, 
            max_len=self.cfg['model']['max_length']
        )
        dataloader = DataLoader(
            dataset, batch_size=self.cfg['training']['batch_size'], 
            shuffle=True, num_workers=0, pin_memory=True
        )
        
        # 针对新层设置大一点的学习率，针对 BERT 设置小一点的学习率
        optimizer = AdamW([
            {'params': self.model.backbone.parameters(), 'lr': self.cfg['training']['learning_rate']},
            {'params': self.model.adapter.parameters(), 'lr': 1e-4}, # Adapter 学习率稍大
            {'params': [self.model.gate], 'lr': 1e-3} # Gate 学习率大，以便快速调整
        ])
        
        loss_fn = SmartAnglELoss(tau=0.05, angle_w=1.0)
        scaler = GradScaler()
        
        total_steps = len(dataloader) * self.cfg['training']['num_epochs'] // self.cfg['training']['gradient_accumulation_steps']
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=total_steps)

        self.model.train()
        accum_steps = self.cfg['training']['gradient_accumulation_steps']
        
        for epoch in range(self.cfg['training']['num_epochs']):
            loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            
            for step, batch in enumerate(loop):
                q_in = {k: v.to(self.device) for k, v in batch['query'].items()}
                p_in = {k: v.to(self.device) for k, v in batch['positive'].items()}
                n_in = {k: v.to(self.device) for k, v in batch['negative'].items()}
                
                with autocast():
                    q_emb = self.model(**q_in)
                    p_emb = self.model(**p_in)
                    n_emb = self.model(**n_in)
                    loss = loss_fn(q_emb, p_emb, n_emb)
                    loss = loss / accum_steps

                scaler.scale(loss).backward()
                
                if (step + 1) % accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # 显示 Gate 的值，看看模型是否启用了 Adapter
                gate_val = self.model.gate.item()
                loop.set_postfix(loss=f"{loss.item()*accum_steps:.4f}", gate=f"{gate_val:.3f}")

            # 保存模型
            save_dir = os.path.join(self.cfg['data']['output_dir'], f"checkpoint-v4-epoch-{epoch+1}")
            if not os.path.exists(save_dir): 
                os.makedirs(save_dir)
            
            # 1. 强制垃圾回收，释放内存
            gc.collect()
            torch.cuda.empty_cache()

            # 2. 关键修改：设置 safe_serialization=False
            # 这会生成 pytorch_model.bin 而不是 model.safetensors，避免 tobytes() 内存爆炸
            try:
                self.model.backbone.save_pretrained(save_dir, safe_serialization=False)
                self.tokenizer.save_pretrained(save_dir)
                print(f"✅ Saved to {save_dir} (using pytorch_model.bin)")
            except Exception as e:
                print(f"❌ Save failed: {e}")
                # 如果还是失败，尝试只保存 state_dict (最省内存的方式)
                torch.save(self.model.backbone.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
                self.tokenizer.save_pretrained(save_dir)
                print(f"✅ Saved using fallback torch.save")

if __name__ == "__main__":
    config_file = r"train_pro\v4\config\train_config.json"
    if os.path.exists(config_file):
        trainer = AnglETrainer(config_file)
        trainer.train()