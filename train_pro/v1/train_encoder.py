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
import numpy as np
from tqdm import tqdm

# --- 设置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. 核心修正：残差复数适配器 (Residual Complex Adapter) ---
class AnglE_Res(nn.Module):
    def __init__(self, model_name, pooling_strategy='cls'):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        self.hidden_size = self.backbone.config.hidden_size
        
        # 【修正点1】改为学习“增量”和“虚部”
        # 我们希望实部保留原始 BERT 语义，只做微调
        # 虚部从零开始学习
        self.adapter = nn.Linear(self.hidden_size, self.hidden_size * 2)
        
        # 【修正点2】零初始化 (Zero Initialization)
        # 关键！将权重和偏置初始化为 0
        # 这样在 step 0，adapter 的输出全是 0
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

    def pooling(self, last_hidden_state, attention_mask):
        if self.pooling_strategy == 'cls':
            return last_hidden_state[:, 0]
        else:
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 1. 获取 BERT 原始输出
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = self.pooling(outputs.last_hidden_state, attention_mask) # [Batch, 768]
        
        # 2. 计算残差项 (初始为 0)
        delta = self.adapter(pooled) # [Batch, 1536]
        
        delta_real = delta[:, :self.hidden_size] # 实部修正量
        delta_imag = delta[:, self.hidden_size:] # 虚部生成量
        
        # 3. 融合：实部 = BERT原值 + 修正量
        # 这样保证了基础能力不丢失
        real_part = pooled + delta_real
        imag_part = delta_imag # 虚部允许从0开始学，或者也可以 pooled + delta_imag
        
        # 4. 拼接
        return torch.cat([real_part, imag_part], dim=1)

# --- 2. 修正后的损失函数 (增加稳定性) ---
class StableAngleLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_cos = config['loss'].get('cosine_w', 0.0)
        self.base_w_angle = config['loss'].get('angle_w', 1.0)
        self.w_ibn = config['loss'].get('ibn_w', 1.0) # 建议开启 IBN
        self.tau = config['loss'].get('cosine_tau', 20.0)
        
    def forward(self, y_pred, y_true):
        # y_pred: [Batch, 1536]
        hidden_dim = y_pred.shape[1] // 2
        real = y_pred[:, :hidden_dim]
        imag = y_pred[:, hidden_dim:]
        
        # 构建复数
        z = real + 1j * imag
        
        # 提取 Pair
        idxs = torch.arange(0, y_pred.shape[0], 2, device=y_pred.device)
        z1 = z[idxs]
        z2 = z[idxs+1]
        labels = y_true[idxs]
        
        # --- A. 计算 Cosine Similarity (用于动态权重) ---
        # 拼接实虚部计算模长归一化后的 Cosine
        vec1 = torch.cat([real[idxs], imag[idxs]], dim=1)
        vec2 = torch.cat([real[idxs+1], imag[idxs+1]], dim=1)
        cos_sim = F.cosine_similarity(vec1, vec2)
        
        # --- B. Angle Loss (优化复数角度) ---
        # z1/z2 的角度
        conjugate_z2 = torch.conj(z2)
        denominator = torch.abs(z2)**2 + 1e-9
        division = (z1 * conjugate_z2) / denominator
        angle_diff = torch.angle(division) # [-pi, pi]
        
        # 动态权重：只在正样本(label=1)且处于饱和区时，加大惩罚
        # 限制最大权重倍数为 3.0，防止梯度爆炸
        saturation = torch.abs(cos_sim).detach()
        dynamic_w = self.base_w_angle * (1 + 2 * saturation)
        
        # 确保维度匹配 [Batch] -> [Batch, 1]
        dynamic_w = dynamic_w.unsqueeze(1)
        labels_exp = labels.unsqueeze(1)
        
        # Loss Calculation
        loss_angle = (dynamic_w * (angle_diff ** 2) * labels_exp).mean()
        
        # --- C. In-Batch Negative (IBN) ---
        # 简单的 InfoNCE Loss 思想：拉近正样本，推开 Batch 内的其他样本
        loss_ibn = 0
        if self.w_ibn > 0:
            # 这里的 cos_sim 已经在前面算过了，但我们需要整个 batch 的矩阵
            # 为了简单，这里只计算 Pair 之间的 Cosine Loss (MSE) 作为辅助
            # 如果你有严格的 IBN 需求，可以使用 CrossEntropy
            # 这里我们用最简单的 MSE 辅助收敛
            loss_ibn = F.mse_loss(cos_sim, labels)

        return loss_angle + self.w_ibn * loss_ibn

# --- 3. 数据集类 (保持不变) ---
class TripletDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_length):
        self.data = raw_data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.data) * 2
    def __getitem__(self, idx):
        pair_idx = idx // 2
        is_second = idx % 2
        item = self.data[pair_idx]
        text = item['text2'] if is_second else item['text1']
        label = float(item['label'])
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {'input_ids': inputs['input_ids'].squeeze(), 'attention_mask': inputs['attention_mask'].squeeze(), 'token_type_ids': inputs['token_type_ids'].squeeze(), 'label': torch.tensor(label, dtype=torch.float)}

# --- 4. 训练器 ---
class AngleResTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f: self.config = json.load(f)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])
        
        # 使用修正后的残差模型
        self.model = AnglE_Res(self.config['model']['name']).to(self.device)
        self.criterion = StableAngleLoss(self.config)

    def train(self):
        # 加载数据 (需确保 utils 路径正确)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
        from data_converter import TripletDataConverter
        raw_data = TripletDataConverter.load_and_convert(self.config['data']['input_jsonl_file'])
        
        dataset = TripletDataset(raw_data, self.tokenizer, self.config['model']['max_length'])
        dataloader = DataLoader(dataset, batch_size=self.config['training']['batch_size'] * 2, shuffle=False, num_workers=0)
        
        # 优化器设置：使用较小的学习率，因为我们是微调
        optimizer = AdamW(self.model.parameters(), lr=1e-5) # 建议从 2e-5 降到 1e-5
        
        num_steps = len(dataloader) * self.config['training']['num_epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_steps)
        
        logger.info("🚀 开始 AnglE-Res (残差版) 训练...")
        self.model.train()
        
        for epoch in range(self.config['training']['num_epochs']):
            total_loss = 0
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            
            for batch in progress:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                emb = self.model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
                
                loss = self.criterion(emb, batch['label'])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # 梯度裁剪，防止爆炸
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                progress.set_postfix({'loss': loss.item()})
            
            # 保存
            save_path = os.path.join(self.config['data']['output_dir'], f"checkpoint-epoch-{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
            self.tokenizer.save_pretrained(save_path)
            self.model.backbone.config.save_pretrained(save_path)
            logger.info(f"💾 模型已保存: {save_path}")

if __name__ == '__main__':
    # 记得改这里为你的 config 路径
    config_path = r"D:\WorkSpace\AnglE_yj\train_pro\v1\config\train_config.json"
    trainer = AngleResTrainer(config_path)
    trainer.train()