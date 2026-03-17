import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import os
import sys
import json
import logging
from tqdm import tqdm

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. AnglE 核心架构 (替代 import angle_emb)
# ==========================================
class AnglE_Standalone(nn.Module):
    """
    完全复刻 AnglE 库的核心模型逻辑：
    Backbone (BERT/RoBERTa) + Pooling (CLS/Mean) + Normalization (Optional)
    """
    def __init__(self, model_name, pooling_strategy='cls', is_llm=False):
        super().__init__()
        self.pooling_strategy = pooling_strategy
        self.is_llm = is_llm
        
        # 加载 HuggingFace Backbone
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # 显存优化：开启梯度检查点 (Gradient Checkpointing)
        # 这对于 AnglE 这种大 Batch Size 训练非常重要
        self.backbone.gradient_checkpointing_enable()

    def pooling(self, last_hidden_state, attention_mask):
        """实现 AnglE 的 Pooling 策略"""
        if self.pooling_strategy == 'cls':
            # 取 [CLS] token 的向量
            return last_hidden_state[:, 0]
        
        elif self.pooling_strategy == 'mean':
            # 计算 Mean Pooling，注意要处理 padding mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.pooling_strategy == 'last':
            # 取最后一个 token (通常用于生成式 LLM，如 LLaMA)
            # 这里简单取最后一个，严谨实现需要根据 padding 找实际最后一个
            return last_hidden_state[:, -1]
        
        else:
            raise ValueError(f"不支持的 Pooling 策略: {self.pooling_strategy}")

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 兼容部分模型没有 token_type_ids 的情况 (如 DistilBERT, RoBERTa 有时不需要)
        model_kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            model_kwargs['token_type_ids'] = token_type_ids

        outputs = self.backbone(**model_kwargs)
        
        # 执行 Pooling
        emb = self.pooling(outputs.last_hidden_state, attention_mask)
        return emb

# ==========================================
# 2. 损失函数 (Smart Angle Loss)
# ==========================================
class SmartAngleLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_angle = config['loss'].get('angle_w', 1.0)
        self.w_ibn = config['loss'].get('ibn_w', 20.0) 
        self.tau = 0.05
        
    def forward(self, y_pred, y_true):
        # y_pred: [Batch, 768]
        batch_size, hidden_dim = y_pred.shape
        half_dim = hidden_dim // 2
        real = y_pred[:, :half_dim]
        imag = y_pred[:, half_dim:]
        
        # 1. 构建复数
        z = real + 1j * imag
        
        # 2. 提取配对 (假设 Batch 是严格配对的: 0-1, 2-3...)
        idxs = torch.arange(0, batch_size, 2, device=y_pred.device)
        z1 = z[idxs]
        z2 = z[idxs+1]
        
        # --- A. Cosine 计算 (用于饱和度检测) ---
        vec1 = torch.cat([real[idxs], imag[idxs]], dim=1)
        vec2 = torch.cat([real[idxs+1], imag[idxs+1]], dim=1)
        cos_sim = F.cosine_similarity(vec1, vec2) 
        
        # --- B. Smart Angle Loss ---
        conjugate_z2 = torch.conj(z2)
        denominator = torch.abs(z2)**2 + 1e-9
        division = (z1 * conjugate_z2) / denominator
        angle_diff = torch.abs(torch.angle(division)) 
        
        # 动态权重
        with torch.no_grad():
            saturation = torch.abs(cos_sim)
            dynamic_scale = 1.0 + (saturation ** 2) 
        dynamic_scale = dynamic_scale.unsqueeze(1)
        
        # 正样本 Mask
        labels_pos = (y_true[idxs] > 0.5).float().unsqueeze(1)
        
        loss_angle = (self.w_angle * dynamic_scale * (angle_diff ** 2) * labels_pos).mean()
        
        # --- C. In-Batch Negative (IBN) ---
        z_norm = F.normalize(y_pred, p=2, dim=1)
        sim_matrix = torch.matmul(z_norm, z_norm.T) / self.tau
        
        # 构造 IBN 标签
        labels_ibn = torch.arange(batch_size, device=y_pred.device)
        labels_ibn = (labels_ibn + 1) - 2 * (labels_ibn % 2)
        
        loss_ibn = F.cross_entropy(sim_matrix, labels_ibn)
        
        return loss_angle + self.w_ibn * loss_ibn

# ==========================================
# 3. 数据集 (配合 DataConverter 使用)
# ==========================================
class TripletDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_length):
        self.data = raw_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self): 
        # TripletDataConverter 通常返回 N 对数据
        # 我们需要把它展开成 2*N 个样本给 DataLoader
        return len(self.data) * 2
        
    def __getitem__(self, idx):
        pair_idx = idx // 2
        is_second = idx % 2
        
        item = self.data[pair_idx]
        
        # 这里兼容 TripletDataConverter 的常见返回格式
        # 假设 item 是 {'text1':..., 'text2':..., 'label':...}
        text = item['text2'] if is_second else item['text1']
        label = float(item['label'])
        
        inputs = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            # 部分模型可能没有 token_type_ids，这里做个兼容
            'token_type_ids': inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

# ==========================================
# 4. 训练器
# ==========================================
class AngleStandaloneTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f: 
            self.config = json.load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化 Tokenizer 和 Standalone AnglE 模型
        model_name = self.config['model']['name']
        pooling = self.config['model'].get('pooling_strategy', 'cls')
        
        logger.info(f"正在初始化模型: {model_name} (Pooling: {pooling})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AnglE_Standalone(model_name, pooling_strategy=pooling).to(self.device)
        
        self.criterion = SmartAngleLoss(self.config)

    def train(self):
        # --- 动态加载 data_converter ---
        # 假设 utils 文件夹在当前脚本的同级或子级目录
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
            from data_converter import TripletDataConverter
            logger.info("成功加载 utils.data_converter")
        except ImportError:
            # 如果找不到，尝试直接从当前目录导入（兼容性处理）
            try:
                from utils.data_converter import TripletDataConverter
            except ImportError:
                raise ImportError("无法找到 `utils.data_converter`。请确保 `utils` 文件夹在 Python 路径下。")
        
        # 1. 加载数据 (使用你的转换器)
        raw_data = TripletDataConverter.load_and_convert(self.config['data']['input_jsonl_file'])
        logger.info(f"加载数据完成，共 {len(raw_data)} 对样本。")
        
        dataset = TripletDataset(
            raw_data, 
            self.tokenizer, 
            self.config['model']['max_length']
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config['training']['batch_size'] * 2, 
            shuffle=False, # 必须为 False，保证成对输入
            num_workers=0
        )
        
        # 优化器
        optimizer = AdamW(self.model.parameters(), lr=self.config['training']['learning_rate'])
        
        # 学习率调度
        accum_steps = self.config['training']['gradient_accumulation_steps']
        total_steps = len(dataloader) // accum_steps * self.config['training']['num_epochs']
        warmup_steps = self.config['training']['warmup_steps']
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        logger.info(f"🚀 开始训练 | Batch: {self.config['training']['batch_size']} | Steps: {total_steps}")
        
        self.model.train()
        optimizer.zero_grad()
        global_step = 0
        
        for epoch in range(self.config['training']['num_epochs']):
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            total_loss = 0
            
            for step, batch in enumerate(progress):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播 (调用 Standalone 模型)
                emb = self.model(
                    batch['input_ids'], 
                    batch['attention_mask'], 
                    batch['token_type_ids']
                )
                
                loss = self.criterion(emb, batch['label'])
                
                loss = loss / accum_steps
                loss.backward()
                
                if (step + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                current_loss = loss.item() * accum_steps
                total_loss += current_loss
                progress.set_postfix({'loss': f"{current_loss:.4f}"})
            
            # 保存
            save_path = os.path.join(self.config['data']['output_dir'], f"checkpoint-epoch-{epoch+1}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            # 保存模型 (Backbone + Config)
            self.model.backbone.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"💾 模型已保存至: {save_path}")

if __name__ == '__main__':
    # 你的配置文件路径
    config_path = r"D:\WorkSpace\AnglE_yj\train_pro\v3\config\train_config.json"
    
    if os.path.exists(config_path):
        trainer = AngleStandaloneTrainer(config_path)
        trainer.train()
    else:
        logger.error(f"找不到配置文件: {config_path}")