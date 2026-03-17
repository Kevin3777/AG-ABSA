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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. 模型回归：标准 AnglE 结构 (无额外参数) ---
class AnglE_Final(nn.Module):
    def __init__(self, model_name, pooling_strategy='cls'):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        
        # 【关键回归】去掉所有 Linear 层！
        # 我们相信 BERT 原本的特征就是最好的，不需要再映射了。

    def pooling(self, last_hidden_state, attention_mask):
        if self.pooling_strategy == 'cls':
            return last_hidden_state[:, 0]
        else:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = self.pooling(outputs.last_hidden_state, attention_mask)
        return pooled # 直接返回 768 维向量

# --- 2. 核心创新：Smart Angle Loss (智能动态损失) ---
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
        
        # 构建复数
        z = real + 1j * imag
        
        # 提取配对
        idxs = torch.arange(0, batch_size, 2, device=y_pred.device)
        z1 = z[idxs]
        z2 = z[idxs+1]
        
        # --- A. 计算 Cosine Similarity ---
        vec1 = torch.cat([real[idxs], imag[idxs]], dim=1)
        vec2 = torch.cat([real[idxs+1], imag[idxs+1]], dim=1)
        cos_sim = F.cosine_similarity(vec1, vec2)
        
        # --- B. 智能角度损失 ---
        conjugate_z2 = torch.conj(z2)
        denominator = torch.abs(z2)**2 + 1e-9
        division = (z1 * conjugate_z2) / denominator
        angle_diff = torch.abs(torch.angle(division)) # [Batch/2, 384]
        
        # 【创新点】饱和度感知加权
        with torch.no_grad():
            saturation = torch.abs(cos_sim)
            dynamic_scale = 1.0 + (saturation ** 2) # [Batch/2]
        
        # 正样本标记
        labels_pos = (y_true[idxs] > 0.5).float() # [Batch/2]

        # 【修复点在这里】：增加维度 [Batch] -> [Batch, 1] 以便广播
        dynamic_scale = dynamic_scale.unsqueeze(1)
        labels_pos = labels_pos.unsqueeze(1)
        
        # 现在: [32, 1] * [32, 384] * [32, 1] -> [32, 384] 正常运算
        loss_angle = (self.w_angle * dynamic_scale * (angle_diff ** 2) * labels_pos).mean()
        
        # --- C. In-Batch Negative (IBN) ---
        z_norm = F.normalize(y_pred, p=2, dim=1)
        sim_matrix = torch.matmul(z_norm, z_norm.T) / self.tau
        
        labels_ibn = torch.arange(batch_size, device=y_pred.device)
        labels_ibn = (labels_ibn + 1) - 2 * (labels_ibn % 2)
        
        loss_ibn = F.cross_entropy(sim_matrix, labels_ibn)
        
        return loss_angle + self.w_ibn * loss_ibn

# --- 3. 数据集与训练器 (标准流程) ---
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

class AngleFinalTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f: self.config = json.load(f)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])
        self.model = AnglE_Final(self.config['model']['name']).to(self.device)
        self.criterion = SmartAngleLoss(self.config)

    def train(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
        from data_converter import TripletDataConverter
        raw_data = TripletDataConverter.load_and_convert(self.config['data']['input_jsonl_file'])
        
        dataset = TripletDataset(raw_data, self.tokenizer, self.config['model']['max_length'])
        dataloader = DataLoader(dataset, batch_size=self.config['training']['batch_size'] * 2, shuffle=False) # 必须False
        
        # 学习率可以稍微回调一点，因为现在没有随机层了
        optimizer = AdamW(self.model.parameters(), lr=2e-5) 
        
        num_steps = len(dataloader) * self.config['training']['num_epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_steps)
        
        logger.info("🚀 开始 AnglE-Final (Smart Loss版) 训练...")
        self.model.train()
        
        for epoch in range(self.config['training']['num_epochs']):
            total_loss = 0
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for batch in progress:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                emb = self.model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
                loss = self.criterion(emb, batch['label']) # Label这里其实只用于AngleLoss判断正样本
                loss.backward()
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                total_loss += loss.item()
                progress.set_postfix({'loss': loss.item()})
            
            save_path = os.path.join(self.config['data']['output_dir'], f"checkpoint-epoch-{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            self.model.backbone.save_pretrained(save_path) # 直接保存 backbone
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"💾 模型已保存: {save_path}")

if __name__ == '__main__':
    # ⚠️ 请确保 config 中的 ibn_w 设置为 1.0 或更高
    config_path = r"D:\WorkSpace\AnglE_yj\train_pro\v2\config\train_config.json"
    trainer = AngleFinalTrainer(config_path)
    trainer.train()