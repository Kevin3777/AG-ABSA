import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
import os
import sys
import gc
import json
import logging
import random
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 使用相对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
angle_path = os.path.join(current_dir, '..', 'AnglE')
sys.path.insert(0, os.path.abspath(angle_path))

# 添加工具路径
utils_path = os.path.join(current_dir, 'utils')
sys.path.insert(0, utils_path)

# 导入必要的库
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from data_converter import TripletDataConverter

class DotProductLoss(nn.Module):
    """点积损失函数（对比学习损失）"""
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        """
        计算点积对比损失
        anchor: 锚点样本
        positive: 正样本
        negative: 负样本
        """
        # 归一化向量（重要：确保是单位向量）
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=-1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=-1)
        
        # 计算余弦相似度（等价于归一化后的点积）
        pos_sim = torch.sum(anchor_embeddings * positive_embeddings, dim=-1) / self.temperature
        neg_sim = torch.sum(anchor_embeddings * negative_embeddings, dim=-1) / self.temperature
        
        # 构建logits矩阵 [batch_size, 2]
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        
        # 标签：正样本（索引0）
        labels = torch.zeros(anchor_embeddings.size(0), dtype=torch.long, device=anchor_embeddings.device)
        
        # 计算交叉熵损失
        loss = self.cross_entropy(logits, labels)
        
        # 计算准确率
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean()
        
        return loss, accuracy

class DotProductEncoder(nn.Module):
    """使用点积损失的编码器模型"""
    def __init__(self, model_name, pooling_strategy='cls', max_length=192):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length
        
        # 获取隐藏层维度
        self.hidden_size = self.model.config.hidden_size
        
    def encode(self, texts, to_numpy=True):
        """编码文本为向量（推理模式）"""
        self.model.eval()
        with torch.no_grad():
            embeddings = self._encode_batch(texts, training=False)
            if to_numpy:
                embeddings = embeddings.cpu().numpy()
        return embeddings
    
    def _encode_batch(self, texts, training=True):
        """批量编码文本"""
        # 对输入进行编码
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 将输入移到设备上
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 获取模型输出
        if training:
            outputs = self.model(**inputs)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)
        
        last_hidden_state = outputs.last_hidden_state
        
        # 根据池化策略获取句子表示
        if self.pooling_strategy == 'cls':
            embeddings = last_hidden_state[:, 0, :]  # CLS token
        elif self.pooling_strategy == 'mean':
            attention_mask = inputs['attention_mask']
            token_embeddings = last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        else:
            raise ValueError(f"不支持的池化策略: {self.pooling_strategy}")
        
        return embeddings
    
    def forward(self, texts):
        """前向传播（用于训练）"""
        return self._encode_batch(texts, training=self.training)

class CheckpointManager:
    """检查点管理器"""
    
    @staticmethod
    def get_latest_checkpoint(checkpoint_dir):
        """获取最新的检查点"""
        if not os.path.exists(checkpoint_dir):
            return None
        
        # 查找所有检查点文件夹
        checkpoints = []
        for item in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, item)):
                if item.startswith("epoch_") or item in ["best_model", "final"]:
                    checkpoints.append(item)
        
        if not checkpoints:
            return None
        
        # 按修改时间排序
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
        
        # 返回最新的检查点（排除best_model和final，因为它们不是训练中间状态）
        for checkpoint in checkpoints:
            if checkpoint.startswith("epoch_"):
                return os.path.join(checkpoint_dir, checkpoint)
        
        return None
    
    @staticmethod
    def save_checkpoint(epoch, model, optimizer, scheduler, scaler, loss, output_dir, is_best=False):
        """保存检查点"""
        checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型
        model.model.save_pretrained(checkpoint_dir)
        model.tokenizer.save_pretrained(checkpoint_dir)
        
        # 保存训练状态
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        
        if scaler is not None:
            checkpoint_state['scaler_state_dict'] = scaler.state_dict()
        
        checkpoint_path = os.path.join(checkpoint_dir, "training_state.pt")
        torch.save(checkpoint_state, checkpoint_path)
        
        # 保存训练配置
        config_path = os.path.join(checkpoint_dir, "training_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'epoch': epoch,
                'loss': loss,
                'timestamp': checkpoint_state['timestamp']
            }, f, indent=2)
        
        logger.info(f"💾 检查点已保存至: {checkpoint_dir} (Epoch {epoch}, Loss: {loss:.4f})")
        
        # 如果是最好模型，额外保存一份
        if is_best:
            best_dir = os.path.join(output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            
            # 复制模型文件
            import shutil
            for file in os.listdir(checkpoint_dir):
                if file.endswith(('.bin', '.json', '.txt', '.model')):
                    shutil.copy2(
                        os.path.join(checkpoint_dir, file),
                        os.path.join(best_dir, file)
                    )
            
            # 保存训练状态
            torch.save(checkpoint_state, os.path.join(best_dir, "training_state.pt"))
            logger.info(f"🏆 最佳模型已保存至: {best_dir}")
    
    @staticmethod
    def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
        
        # 加载训练状态
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(state_path):
            checkpoint = torch.load(state_path, map_location='cpu')
            
            # 加载模型
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载调度器状态
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 加载scaler状态
            if scaler is not None and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', float('inf'))
            
            logger.info(f"🔄 从检查点恢复: {checkpoint_path}")
            logger.info(f"  恢复的Epoch: {epoch}, 损失: {loss:.4f}")
            
            return epoch, loss
        else:
            # 如果没有训练状态文件，只加载模型权重
            model.model = AutoModel.from_pretrained(checkpoint_path)
            model.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            logger.info(f"🤖 加载模型权重: {checkpoint_path}")
            return 0, float('inf')

class DotProductAngleTrainer:
    def __init__(self, config_path="train_dotproduct/config/dot_product_config.json", resume=False):
        self.config = self.load_config(config_path)
        self.resume = resume  # 是否从检查点恢复
        
        # 确保输出目录不同
        self.config['data']['output_dir'] = "checkpoint_dotproduct"
        
        # 初始化损失函数
        self.loss_fn = DotProductLoss(
            temperature=self.config['loss'].get('temperature', 0.05)
        )
        
        # 初始化检查点管理器
        self.checkpoint_manager = CheckpointManager()
        
        # 设置随机种子
        self.set_seed(42)
    
    def set_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✅ 点积训练配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"⚠️ 配置文件加载失败，使用默认配置: {e}")
            return self.create_default_config()
    
    def create_default_config(self):
        """创建默认配置（与原配置保持一致）"""
        config = {
            'data': {
                'input_jsonl_file': 'D:/WorkSpace/AnglE_yj/data_preparation/Aspect-Polarity_Pair/output/v2/asap_angle_contextual_ap_data_hybrid.jsonl',
                'output_dir': 'checkpoint_dotproduct'
            },
            'model': {
                'name': 'hfl/chinese-roberta-wwm-ext',
                'max_length': 192,
                'pooling_strategy': 'cls'
            },
            'training': {
                'batch_size': 32,
                'gradient_accumulation_steps': 4,
                'num_epochs': 5,
                'learning_rate': 2e-5,
                'weight_decay': 0.01,
                'warmup_steps': 500,
                'save_steps': 1000,
                'logging_steps': 100,
                'fp16': True,
                'resume_from_checkpoint': True  # 添加恢复选项
            },
            'loss': {
                'temperature': 0.05
            }
        }
        logger.info("📋 使用默认点积训练配置")
        return config
    
    def save_config(self, output_dir):
        """保存配置到输出目录"""
        os.makedirs(output_dir, exist_ok=True)
        config_save_path = os.path.join(output_dir, "dot_product_train_config.json")
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 点积训练配置已保存至: {config_save_path}")
    
    def cleanup_memory(self):
        """清理GPU内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - allocated
            logger.info(f"🧹 内存清理完成 - 已分配: {allocated:.2f}GB, 可用: {free:.2f}GB")
    
    def print_gpu_info(self):
        """打印GPU信息"""
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            total_memory = gpu_props.total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            free = total_memory - allocated
            
            logger.info(f"🎮 GPU: {gpu_props.name}")
            logger.info(f"💾 显存: 总共 {total_memory:.1f}GB, 已用 {allocated:.2f}GB, 可用 {free:.2f}GB")
            logger.info(f"🔧 CUDA能力: {gpu_props.major}.{gpu_props.minor}")
        else:
            logger.warning("⚠️ 未检测到CUDA设备，将使用CPU训练")
    
    def load_and_prepare_data(self):
        """加载和准备数据"""
        try:
            input_file = self.config['data']['input_jsonl_file']
            logger.info(f"📥 正在加载数据集: {input_file}")
            
            # 检查文件是否存在
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"数据文件不存在: {input_file}")
            
            # 使用数据转换器加载和转换数据
            dataset = TripletDataConverter.load_and_convert(input_file)
            
            # 确保数据格式正确
            processed_data = []
            for item in dataset:
                # 检查数据格式
                if 'text1' in item and 'text2' in item and 'label' in item:
                    processed_data.append(item)
                else:
                    logger.warning(f"⚠️ 跳过格式错误的数据: {item}")
            
            logger.info(f"📊 数据统计: 总共 {len(processed_data)} 个样本")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            raise
    
    def prepare_triplet_batch(self, batch_data):
        """准备三元组批次数据"""
        texts1 = [item['text1'] for item in batch_data]
        texts2 = [item['text2'] for item in batch_data]
        labels = [item['label'] for item in batch_data]
        
        # 根据标签分离正负样本
        anchor_texts = []
        positive_texts = []
        negative_texts = []
        
        for i, label in enumerate(labels):
            if label == 1:  # text2是正样本
                anchor_texts.append(texts1[i])
                positive_texts.append(texts2[i])
                # 从同批次中随机选择一个负样本
                negative_idx = i
                while negative_idx == i:
                    negative_idx = random.randint(0, len(batch_data) - 1)
                negative_texts.append(texts2[negative_idx])
            else:  # label == 0, text2是负样本
                anchor_texts.append(texts1[i])
                # 对于负样本对，我们使用text1自身作为正样本
                positive_texts.append(texts1[i])
                negative_texts.append(texts2[i])
        
        return anchor_texts, positive_texts, negative_texts
    
    def train_epoch(self, model, dataloader, optimizer, scheduler, device, scaler=None, epoch_idx=0, start_batch=0):
        """训练一个epoch"""
        model.train()  # 确保模型在训练模式
        total_loss = 0
        total_acc = 0
        total_batches = len(dataloader)
        
        # 如果从中间恢复，跳过已经训练过的batch
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx + 1}", leave=False)
        for batch_idx, batch_data in enumerate(pbar):
            # 如果恢复训练，跳过已经训练过的batch
            if batch_idx < start_batch:
                continue
            
            # 准备三元组数据
            anchor_texts, positive_texts, negative_texts = self.prepare_triplet_batch(batch_data)
            
            # 编码锚点、正样本、负样本
            if scaler is not None:
                with torch.amp.autocast('cuda', enabled=True):
                    anchor_embeddings = model(anchor_texts)
                    positive_embeddings = model(positive_texts)
                    negative_embeddings = model(negative_texts)
                    
                    # 计算损失
                    loss, accuracy = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            else:
                anchor_embeddings = model(anchor_texts)
                positive_embeddings = model(positive_texts)
                negative_embeddings = model(negative_texts)
                
                # 计算损失
                loss, accuracy = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            
            # 梯度累积
            loss = loss / self.config['training']['gradient_accumulation_steps']
            
            # 反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
                
                # 梯度累积步骤
                if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
            else:
                loss.backward()
                
                # 梯度累积步骤
                if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
            
            # 统计信息
            total_loss += loss.item() * self.config['training']['gradient_accumulation_steps']
            total_acc += accuracy.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item() * self.config['training']['gradient_accumulation_steps']:.4f}",
                'acc': f"{accuracy.item():.4f}"
            })
            
            # 定期保存检查点
            save_steps = self.config['training']['save_steps']
            if save_steps > 0 and (batch_idx + 1) % save_steps == 0:
                current_loss = loss.item() * self.config['training']['gradient_accumulation_steps']
                self.checkpoint_manager.save_checkpoint(
                    epoch_idx + 1, model, optimizer, scheduler, scaler, current_loss,
                    os.path.join(self.config['data']['output_dir'], "dot_product_model")
                )
            
            # 日志记录
            if (batch_idx + 1) % self.config['training']['logging_steps'] == 0:
                logger.info(f"  📊 Batch {batch_idx + 1}/{total_batches} - "
                          f"Loss: {loss.item() * self.config['training']['gradient_accumulation_steps']:.4f}, "
                          f"Acc: {accuracy.item():.4f}")
        
        pbar.close()
        
        # 计算平均指标
        processed_batches = total_batches - start_batch
        avg_loss = total_loss / processed_batches if processed_batches > 0 else 0
        avg_acc = total_acc / processed_batches if processed_batches > 0 else 0
        
        return avg_loss, avg_acc
    
    def create_dataloader(self, data, batch_size, shuffle=True):
        """创建数据加载器"""
        class SimpleDataLoader:
            def __init__(self, data, batch_size, shuffle=True):
                self.data = data
                self.batch_size = batch_size
                self.shuffle = shuffle
                self.num_batches = (len(data) + batch_size - 1) // batch_size
            
            def __iter__(self):
                if self.shuffle:
                    random.shuffle(self.data)
                self.current_idx = 0
                return self
            
            def __next__(self):
                if self.current_idx >= len(self.data):
                    raise StopIteration
                
                end_idx = min(self.current_idx + self.batch_size, len(self.data))
                batch = self.data[self.current_idx:end_idx]
                self.current_idx = end_idx
                
                return batch
            
            def __len__(self):
                return self.num_batches
        
        return SimpleDataLoader(data, batch_size, shuffle)
    
    def train(self):
        """主训练函数 - 使用点积损失"""
        logger.info("🚀 开始点积损失模型训练")
        logger.info("📌 输出目录: checkpoint2")
        logger.info(f"🔄 恢复训练: {self.resume}")
        
        # 显示配置
        logger.info("📋 点积训练配置:")
        logger.info(f"  模型: {self.config['model']['name']}")
        logger.info(f"  Batch Size: {self.config['training']['batch_size']}")
        logger.info(f"  梯度累积: {self.config['training']['gradient_accumulation_steps']}")
        logger.info(f"  有效Batch Size: {self.config['training']['batch_size'] * self.config['training']['gradient_accumulation_steps']}")
        logger.info(f"  序列长度: {self.config['model']['max_length']}")
        logger.info(f"  学习率: {self.config['training']['learning_rate']}")
        logger.info(f"  训练轮数: {self.config['training']['num_epochs']}")
        logger.info(f"  温度参数: {self.config['loss'].get('temperature', 0.05)}")
        
        # 清理内存并显示GPU信息
        self.cleanup_memory()
        self.print_gpu_info()
        
        # 1. 加载模型和分词器
        try:
            logger.info("🤖 正在加载模型...")
            model = DotProductEncoder(
                model_name=self.config['model']['name'],
                pooling_strategy=self.config['model']['pooling_strategy'],
                max_length=self.config['model']['max_length']
            )
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            # 确保模型参数需要梯度
            for param in model.parameters():
                param.requires_grad = True
            
            if device.type == 'cuda':
                logger.info("✅ 模型已转移到GPU，梯度计算已启用")
            else:
                logger.warning("⚠️ 使用CPU进行训练，速度会较慢")
                
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return
        
        # 2. 创建优化器
        optimizer = AdamW(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0.01)
        )
        
        # 3. 创建学习率调度器
        total_steps = 0
        scheduler = None
        warmup_steps = self.config['training']['warmup_steps']
        
        # 4. 创建混合精度训练的scaler
        scaler = None
        if self.config['training']['fp16'] and torch.cuda.is_available():
            scaler = torch.amp.GradScaler('cuda')
            logger.info("🔧 启用FP16混合精度训练")
        else:
            logger.info("🔧 禁用FP16")
        
        # 5. 加载数据
        try:
            train_data = self.load_and_prepare_data()
        except Exception as e:
            logger.error(f"❌ 数据准备失败: {e}")
            return
        
        # 6. 创建数据加载器
        dataloader = self.create_dataloader(
            train_data, 
            self.config['training']['batch_size'], 
            shuffle=True
        )
        
        # 7. 检查点恢复
        start_epoch = 0
        start_batch = 0
        best_loss = float('inf')
        
        if self.resume:
            checkpoint_dir = os.path.join(self.config['data']['output_dir'], "dot_product_model")
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint(checkpoint_dir)
            
            if latest_checkpoint:
                try:
                    start_epoch, best_loss = self.checkpoint_manager.load_checkpoint(
                        latest_checkpoint, model, optimizer, scheduler, scaler
                    )
                    
                    # 解析epoch信息
                    checkpoint_name = os.path.basename(latest_checkpoint)
                    if checkpoint_name.startswith("epoch_"):
                        start_epoch = int(checkpoint_name.split("_")[1])
                    
                    logger.info(f"✅ 从检查点恢复成功: Epoch {start_epoch}, Loss {best_loss:.4f}")
                    
                    # 重新计算总步数
                    total_steps = len(dataloader) * (self.config['training']['num_epochs'] - start_epoch)
                    
                    # 重新创建调度器（如果需要）
                    if warmup_steps > 0:
                        scheduler = get_linear_schedule_with_warmup(
                            optimizer,
                            num_warmup_steps=warmup_steps,
                            num_training_steps=total_steps
                        )
                        logger.info(f"🔄 重新创建学习率调度器，剩余训练步数: {total_steps}")
                    
                except Exception as e:
                    logger.error(f"❌ 检查点恢复失败: {e}")
                    logger.info("⚠️ 将从头开始训练")
                    start_epoch = 0
            else:
                logger.info("ℹ️ 未找到检查点，将从头开始训练")
        
        # 如果没有恢复或恢复失败，从头开始
        if start_epoch == 0:
            total_steps = len(dataloader) * self.config['training']['num_epochs']
            if warmup_steps > 0:
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
        
        # 8. 保存配置
        output_dir = self.config['data']['output_dir']
        self.save_config(output_dir)
        
        # 9. 创建保存目录
        model_save_dir = os.path.join(output_dir, "dot_product_model")
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 10. 开始训练
        logger.info("🎯 开始模型训练...")
        
        for epoch in range(start_epoch, self.config['training']['num_epochs']):
            logger.info(f"\n🔄 Epoch {epoch + 1}/{self.config['training']['num_epochs']}")
            
            try:
                epoch_loss, epoch_acc = self.train_epoch(
                    model, dataloader, optimizer, scheduler, device, scaler, epoch, start_batch
                )
                
                # 重置start_batch（只在第一个恢复的epoch需要）
                start_batch = 0
                
                logger.info(f"✅ Epoch {epoch + 1} 完成 - 平均Loss: {epoch_loss:.4f}, 平均Acc: {epoch_acc:.4f}")
                
                # 保存检查点
                self.checkpoint_manager.save_checkpoint(
                    epoch + 1, model, optimizer, scheduler, scaler, epoch_loss, model_save_dir
                )
                
                # 保存最佳模型
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_dir = os.path.join(model_save_dir, "best_model")
                    self.checkpoint_manager.save_checkpoint(
                        epoch + 1, model, optimizer, scheduler, scaler, epoch_loss, model_save_dir, is_best=True
                    )
                    logger.info(f"🏆 保存最佳模型 (Loss: {epoch_loss:.4f})")
                
                # 定期保存检查点
                save_steps = self.config['training']['save_steps']
                if save_steps > 0 and (epoch + 1) % 2 == 0:  # 每2个epoch保存一次
                    checkpoint_dir = os.path.join(model_save_dir, f"epoch_{epoch + 1}")
                    self.checkpoint_manager.save_checkpoint(
                        epoch + 1, model, optimizer, scheduler, scaler, epoch_loss, model_save_dir
                    )
                
                # 清理内存
                self.cleanup_memory()
                
            except torch.cuda.OutOfMemoryError:
                logger.error("💥 GPU内存不足！")
                self.handle_oom_error()
                break
            except Exception as e:
                logger.error(f"💥 Epoch {epoch + 1} 训练失败: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # 11. 保存最终模型
        final_model_dir = os.path.join(model_save_dir, "final")
        self.checkpoint_manager.save_checkpoint(
            self.config['training']['num_epochs'], model, optimizer, scheduler, scaler,
            best_loss, model_save_dir
        )
        
        logger.info(f"\n🎉 点积损失训练完成！")
        logger.info(f"📁 模型目录: {model_save_dir}")
        logger.info(f"🏆 最佳模型: {os.path.join(model_save_dir, 'best_model')}")
        logger.info(f"🎯 最终模型: {final_model_dir}")
        logger.info(f"📊 最佳损失: {best_loss:.4f}")
    
    def handle_oom_error(self):
        """处理内存不足错误"""
        logger.info("💡 内存优化建议:")
        logger.info("  1. 降低 batch_size")
        logger.info("  2. 降低 max_length")
        logger.info("  3. 禁用 FP16")
        logger.info("  4. 使用更小的模型")
        logger.info("  5. 增加 gradient_accumulation_steps")

def main():
    """主函数"""
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='点积损失模型训练')
    parser.add_argument('--config', type=str, default='train_dotproduct/config/dot_product_config.json',
                       help='配置文件路径')
    parser.add_argument('--resume', action='store_true',
                       help='从最近的检查点恢复训练')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='指定检查点路径恢复训练')
    
    args = parser.parse_args()
    
    try:
        # 如果指定了配置文件不存在，创建默认配置
        if not os.path.exists(args.config):
            logger.info("📝 创建点积训练配置文件...")
            os.makedirs(os.path.dirname(args.config), exist_ok=True)
            default_config = {
                "data": {
                    "input_jsonl_file": "D:/WorkSpace/AnglE_yj/data_preparation/Aspect-Polarity_Pair/output/v2/asap_angle_contextual_ap_data_hybrid.jsonl",
                    "output_dir": "checkpoint_dotproduct"
                },
                "model": {
                    "name": "hfl/chinese-roberta-wwm-ext",
                    "max_length": 192,
                    "pooling_strategy": "cls"
                },
                "training": {
                    "batch_size": 32,
                    "gradient_accumulation_steps": 4,
                    "num_epochs": 5,
                    "learning_rate": 2e-5,
                    "weight_decay": 0.01,
                    "warmup_steps": 500,
                    "save_steps": 1000,
                    "logging_steps": 100,
                    "fp16": True,
                    "resume_from_checkpoint": True
                },
                "loss": {
                    "temperature": 0.05
                }
            }
            with open(args.config, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ 已创建配置文件: {args.config}")
        
        # 创建训练器
        trainer = DotProductAngleTrainer(config_path=args.config, resume=args.resume)
        trainer.train()
        
    except KeyboardInterrupt:
        logger.info("⏹️ 训练被用户中断")
        logger.info("💾 下次运行可以使用 --resume 参数恢复训练")
    except Exception as e:
        logger.error(f"💥 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()