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

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 模型定义：原始架构 + 显存优化 ---
class AnglE_Final(nn.Module):
    def __init__(self, model_name, pooling_strategy='cls'):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # 【关键优化】开启梯度检查点
        # 这允许我们在显存有限的情况下，使用 Batch Size 32 运行复杂的对比学习 Loss
        self.backbone.gradient_checkpointing_enable()
        
        self.pooling_strategy = pooling_strategy

    def pooling(self, last_hidden_state, attention_mask):
        if self.pooling_strategy == 'cls':
            return last_hidden_state[:, 0]
        else:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 直接输出 768 维向量，不加任何层
        return self.pooling(outputs.last_hidden_state, attention_mask)

# --- 损失函数：Smart Angle Loss (修复版) ---
class SmartAngleLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_angle = config['loss'].get('angle_w', 1.0)
        # IBN 权重设大一点，让 Loss 数值明显
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
        
        # 2. 提取配对 (text1, text2)
        idxs = torch.arange(0, batch_size, 2, device=y_pred.device)
        z1 = z[idxs]
        z2 = z[idxs+1]
        
        # --- A. 计算 Cosine (用于饱和度检测) ---
        vec1 = torch.cat([real[idxs], imag[idxs]], dim=1)
        vec2 = torch.cat([real[idxs+1], imag[idxs+1]], dim=1)
        cos_sim = F.cosine_similarity(vec1, vec2) # [Batch/2]
        
        # --- B. Smart Angle Loss ---
        conjugate_z2 = torch.conj(z2)
        denominator = torch.abs(z2)**2 + 1e-9
        division = (z1 * conjugate_z2) / denominator
        angle_diff = torch.abs(torch.angle(division)) # [Batch/2, 384]
        
        # 动态加权逻辑
        with torch.no_grad():
            saturation = torch.abs(cos_sim)
            # 相似度越高(饱和)，权重越大(1.0 -> 2.0)
            dynamic_scale = 1.0 + (saturation ** 2) 
        
        # 维度修正：[Batch/2] -> [Batch/2, 1] 以支持广播
        dynamic_scale = dynamic_scale.unsqueeze(1)
        
        # 正样本标记
        labels_pos = (y_true[idxs] > 0.5).float().unsqueeze(1)
        
        loss_angle = (self.w_angle * dynamic_scale * (angle_diff ** 2) * labels_pos).mean()
        
        # --- C. In-Batch Negative (IBN) ---
        # 全 Batch 对比学习
        z_norm = F.normalize(y_pred, p=2, dim=1)
        sim_matrix = torch.matmul(z_norm, z_norm.T) / self.tau
        
        # 构造对比标签: 0与1是一对，2与3是一对...
        labels_ibn = torch.arange(batch_size, device=y_pred.device)
        labels_ibn = (labels_ibn + 1) - 2 * (labels_ibn % 2)
        
        loss_ibn = F.cross_entropy(sim_matrix, labels_ibn)
        
        return loss_angle + self.w_ibn * loss_ibn

# --- 数据集类 ---
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

# --- 训练器 ---
class AngleFinalTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f: self.config = json.load(f)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])
        
        # 初始化最终模型
        self.model = AnglE_Final(self.config['model']['name']).to(self.device)
        self.criterion = SmartAngleLoss(self.config)

    def train(self):
        # 动态加载 utils
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
        from data_converter import TripletDataConverter
        
        # 1. 加载数据
        raw_data = TripletDataConverter.load_and_convert(self.config['data']['input_jsonl_file'])
        dataset = TripletDataset(raw_data, self.tokenizer, self.config['model']['max_length'])
        
        # shuffle=False 必须保留，否则 text1 和 text2 无法配对
        dataloader = DataLoader(dataset, batch_size=self.config['training']['batch_size'] * 2, shuffle=False)
        
        # 优化器
        optimizer = AdamW(self.model.parameters(), lr=self.config['training']['learning_rate'])
        
        # 步数计算
        accum_steps = self.config['training']['gradient_accumulation_steps']
        total_steps = len(dataloader) // accum_steps * self.config['training']['num_epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.config['training']['warmup_steps'], num_training_steps=total_steps)
        
        logger.info(f"🚀 开始训练 | Batch: {self.config['training']['batch_size']} | Accum: {accum_steps} | IBN_W: {self.config['loss']['ibn_w']}")
        self.model.train()
        
        global_step = 0
        optimizer.zero_grad()
        
        for epoch in range(self.config['training']['num_epochs']):
            total_loss = 0
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            
            for step, batch in enumerate(progress):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                emb = self.model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
                loss = self.criterion(emb, batch['label'])
                
                # 梯度累积：Loss 除以步数
                loss = loss / accum_steps
                loss.backward()
                
                # 只有达到累积步数才更新
                if (step + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                # 显示用的 Loss (还原回真实值)
                current_loss = loss.item() * accum_steps
                total_loss += current_loss
                progress.set_postfix({'loss': f"{current_loss:.4f}"})
            
            # 保存
            save_path = os.path.join(self.config['data']['output_dir'], f"checkpoint-epoch-{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            self.model.backbone.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"💾 模型已保存: {save_path}")

if __name__ == '__main__':
    # 记得修改 config 路径
    config_path = r"D:\WorkSpace\AnglE_yj\train\v2\config\train_config.json"
    trainer = AngleFinalTrainer(config_path)
    trainer.train()