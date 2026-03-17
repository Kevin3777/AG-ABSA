import torch
import os
import sys
import gc
import json
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 使用相对路径导入data_converter
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, 'utils')
sys.path.insert(0, utils_path)

from data_converter import TripletDataConverter

# ============================================================================
# 1. AnglE 核心实现（简化版）
# ============================================================================

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, Trainer, TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Dict, List, Optional, Union, Any

def get_pooling(outputs: torch.Tensor,
                inputs: Dict,
                pooling_strategy: str,
                padding_side: str) -> torch.Tensor:
    """池化模型输出"""
    if pooling_strategy == 'cls':
        outputs = outputs[:, 0]
    elif pooling_strategy == 'mean':
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / inputs["attention_mask"].sum(dim=1).unsqueeze(1)
    elif pooling_strategy == 'max':
        outputs, _ = torch.max(outputs * inputs["attention_mask"][:, :, None], dim=1)
    else:
        raise NotImplementedError(
            '请从 [`cls`, `mean`, `max`] 中选择池化策略')
    return outputs


def cosine_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float = 20.0) -> torch.Tensor:
    """计算余弦损失"""
    # y_true: (batch_size, 1), y_pred: (batch_size * 2, hidden_size)
    # 我们需要将y_true重塑为(batch_size * 2, 1)
    batch_size = y_true.shape[0]
    y_true = y_true.repeat(1, 2).reshape(-1, 1)
    
    # 只取每对的第一个元素进行比较
    y_true = y_true[::2, 0]
    y_true = (y_true[:, None] < y_true[None, :]).float()
    
    y_pred = F.normalize(y_pred, p=2, dim=1)
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * tau
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = (y_pred - (1 - y_true) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)


def in_batch_negative_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    tau: float = 20.0,
    negative_weights: float = 0.0
) -> torch.Tensor:
    """计算批内负样本损失 - 简化版"""
    device = y_pred.device
    
    # y_true: (batch_size, 1), y_pred: (batch_size * 2, hidden_size)
    batch_size = y_true.shape[0]
    
    # 提取query和positive的特征
    query = y_pred[::2]  # (batch_size, hidden_size)
    positive = y_pred[1::2]  # (batch_size, hidden_size)
    
    # 计算相似度矩阵
    query_norm = F.normalize(query, p=2, dim=1)
    positive_norm = F.normalize(positive, p=2, dim=1)
    
    # 计算query和positive之间的相似度
    similarities = torch.matmul(query_norm, positive_norm.T) * tau  # (batch_size, batch_size)
    
    # 创建标签：对角线位置是正样本
    labels = torch.arange(batch_size, device=device)
    
    # 计算交叉熵损失
    loss = F.cross_entropy(similarities, labels)
    
    return loss


class AngleLoss:
    """AnglE损失函数配置"""
    def __init__(self,
                 cosine_w: float = 0.0,
                 ibn_w: float = 1.0,
                 cln_w: float = 1.0,
                 angle_w: float = 0.02,
                 cosine_tau: float = 20.0,
                 ibn_tau: float = 20.0,
                 angle_tau: float = 20.0,
                 angle_pooling_strategy: str = 'sum',
                 **kwargs):
        self.cosine_w = cosine_w
        self.ibn_w = ibn_w
        self.cln_w = cln_w
        self.angle_w = angle_w
        self.cosine_tau = cosine_tau
        self.ibn_tau = ibn_tau
        self.angle_tau = angle_tau
        self.angle_pooling_strategy = angle_pooling_strategy

    def __call__(self,
                 labels: torch.Tensor,
                 outputs: torch.Tensor) -> torch.Tensor:
        """计算AnglE损失"""
        loss = 0.
        if self.cosine_w > 0:
            loss += self.cosine_w * cosine_loss(labels, outputs, self.cosine_tau)
        if self.ibn_w > 0:
            loss += self.ibn_w * in_batch_negative_loss(labels, outputs, self.ibn_tau)
        return loss


class Pooler:
    """使用Pooler获取句子嵌入"""
    def __init__(self,
                 model: PreTrainedModel,
                 pooling_strategy: Optional[str] = None,
                 padding_side: Optional[str] = None):
        self.model = model
        self.pooling_strategy = pooling_strategy
        self.padding_side = padding_side

    def __call__(self,
                 inputs: Dict,
                 pooling_strategy: Optional[str] = None) -> torch.Tensor:
        """获取句子嵌入"""
        ret = self.model(output_hidden_states=True, **inputs)
        outputs = ret.last_hidden_state if hasattr(ret, 'last_hidden_state') else ret.hidden_states[-1]
        
        outputs = get_pooling(outputs, inputs,
                              pooling_strategy or self.pooling_strategy,
                              padding_side=self.padding_side)
        return outputs


class AnglE(nn.Module):
    """AnglE 模型主类"""
    
    def __init__(self, model_name_or_path, max_length=512, pooling_strategy="cls"):
        super().__init__()
        
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        
        # 获取隐藏层大小
        self.hidden_size = self.model.config.hidden_size
        
        self.pooler = Pooler(
            self.model,
            pooling_strategy=self.pooling_strategy,
            padding_side=self.tokenizer.padding_side)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """前向传播"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        
        # 获取最后一层隐藏状态
        last_hidden_state = outputs.last_hidden_state
        
        # 池化
        pooled_output = get_pooling(last_hidden_state, 
                                   {"attention_mask": attention_mask},
                                   self.pooling_strategy,
                                   self.tokenizer.padding_side)
        
        return pooled_output
    
    @staticmethod
    def from_pretrained(model_name_or_path: str,
                        max_length: int = 512,
                        pooling_strategy: str = 'cls',
                        **kwargs):
        """从预训练模型加载AnglE"""
        angle = AnglE(model_name_or_path,
                      max_length=max_length,
                      pooling_strategy=pooling_strategy,
                      **kwargs)
        return angle
    
    def fit(self, train_ds, output_dir, batch_size=32, epochs=3, learning_rate=2e-5,
            warmup_steps=100, save_steps=500, logging_steps=100, 
            gradient_accumulation_steps=1, fp16=False, loss_kwargs=None, **kwargs):
        """训练模型"""
        # 导入必要的类
        from dataclasses import dataclass
        from typing import List
        import random
        
        @dataclass
        class SimpleAngleDataCollator:
            """简化的数据整理器"""
            tokenizer: PreTrainedTokenizerBase
            max_length: int = 512
            text_prompt: Optional[str] = None
            
            def __call__(self, features: List[Dict]):
                """整理函数"""
                texts1 = []
                texts2 = []
                labels = []
                
                for feature in features:
                    text1 = feature['text1']
                    text2 = feature['text2']
                    label = feature['label']
                    
                    # 应用提示（如果提供）
                    if self.text_prompt is not None:
                        text1 = self.text_prompt.format(text=text1)
                        text2 = self.text_prompt.format(text=text2)
                    
                    texts1.append(text1)
                    texts2.append(text2)
                    labels.append(label)
                
                # 编码第一个文本
                encoding1 = self.tokenizer(
                    texts1,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                # 编码第二个文本
                encoding2 = self.tokenizer(
                    texts2,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids_1': encoding1['input_ids'],
                    'attention_mask_1': encoding1['attention_mask'],
                    'input_ids_2': encoding2['input_ids'],
                    'attention_mask_2': encoding2['attention_mask'],
                    'labels': torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
                }
        
        class SimpleAngleTrainer(Trainer):
            """简化的Angle训练器"""
            def __init__(self, *args, loss_kwargs=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.loss_kwargs = loss_kwargs or {}
                self.loss_fn = AngleLoss(**self.loss_kwargs)
            
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                """计算损失 - 添加**kwargs以处理额外参数"""
                # 获取第一个文本的特征
                outputs1 = model(
                    input_ids=inputs['input_ids_1'],
                    attention_mask=inputs['attention_mask_1']
                )
                
                # 获取第二个文本的特征
                outputs2 = model(
                    input_ids=inputs['input_ids_2'],
                    attention_mask=inputs['attention_mask_2']
                )
                
                # 合并输出
                batch_size = outputs1.shape[0]
                outputs = torch.zeros(batch_size * 2, outputs1.shape[1], device=outputs1.device)
                outputs[0::2] = outputs1
                outputs[1::2] = outputs2
                
                # 计算损失
                loss = self.loss_fn(inputs['labels'], outputs)
                
                return (loss, (outputs1, outputs2)) if return_outputs else loss
        
        # 配置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            fp16=fp16,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=2,
            remove_unused_columns=False,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
        )
        
        # 创建训练器
        trainer = SimpleAngleTrainer(
            model=self,
            args=training_args,
            train_dataset=train_ds,
            data_collator=SimpleAngleDataCollator(
                tokenizer=self.tokenizer,
                max_length=self.max_length
            ),
            loss_kwargs=loss_kwargs
        )
        
        # 开始训练
        trainer.train()
        
        # 保存模型
        self.save_pretrained(output_dir)
        
        logger.info(f"✅ 模型训练完成并保存到: {output_dir}")
    
    def save_pretrained(self, output_dir: str):
        """保存模型"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


# ============================================================================
# 2. 训练器主类
# ============================================================================

class AngleTrainerWrapper:
    def __init__(self, config_path=r"train\v2_another\config\train_config.json"):
        self.config = self.load_config(config_path)
        
    def load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✅ 配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败: {e}")
            raise
    
    def save_config(self, output_dir):
        """保存配置到输出目录"""
        os.makedirs(output_dir, exist_ok=True)
        config_save_path = os.path.join(output_dir, "train_config.json")
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 配置已保存至: {config_save_path}")
    
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
        """加载和准备数据 - 使用外部的data_converter"""
        try:
            input_file = self.config['data']['input_jsonl_file']
            logger.info(f"📥 正在加载数据集: {input_file}")
            
            # 检查文件是否存在
            if not os.path.exists(input_file):
                logger.error(f"❌ 数据文件不存在: {input_file}")
                # 尝试使用相对路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                possible_path = os.path.join(current_dir, input_file)
                if os.path.exists(possible_path):
                    input_file = possible_path
                    logger.info(f"📁 使用相对路径: {input_file}")
                else:
                    raise FileNotFoundError(f"数据文件不存在: {input_file}")
            
            # 使用数据转换器加载和转换数据
            dataset = TripletDataConverter.load_and_convert(input_file)
            
            # 显示样本示例
            if len(dataset) > 0:
                logger.info("🔍 数据样本示例:")
                for i in range(min(3, len(dataset))):
                    sample = dataset[i]
                    logger.info(f"  样本 {i+1}:")
                    logger.info(f"    text1: {sample['text1'][:80]}...")
                    logger.info(f"    text2: {sample['text2'][:80]}...")
                    logger.info(f"    label: {sample['label']}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            # 打印更多调试信息
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def train(self):
        """主训练函数"""
        logger.info("🚀 开始AnglE模型训练")
        
        # 显示配置
        logger.info("📋 训练配置:")
        logger.info(f"  模型: {self.config['model']['name']}")
        logger.info(f"  Batch Size: {self.config['training']['batch_size']}")
        logger.info(f"  梯度累积: {self.config['training']['gradient_accumulation_steps']}")
        logger.info(f"  有效Batch Size: {self.config['training']['batch_size'] * self.config['training']['gradient_accumulation_steps']}")
        logger.info(f"  序列长度: {self.config['model']['max_length']}")
        logger.info(f"  学习率: {self.config['training']['learning_rate']}")
        logger.info(f"  训练轮数: {self.config['training']['num_epochs']}")
        
        # 清理内存并显示GPU信息
        self.cleanup_memory()
        self.print_gpu_info()
        
        # 1. 加载模型
        try:
            logger.info("🤖 正在加载模型...")
            angle = AnglE.from_pretrained(
                model_name_or_path=self.config['model']['name'],
                max_length=self.config['model']['max_length'],
                pooling_strategy=self.config['model']['pooling_strategy']
            )
            
            if torch.cuda.is_available():
                angle = angle.cuda()
                logger.info("✅ 模型已转移到GPU")
            else:
                logger.warning("⚠️ 使用CPU进行训练，速度会较慢")
                
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return
        
        # 2. 加载数据
        try:
            train_ds = self.load_and_prepare_data()
            logger.info(f"✅ 成功加载 {len(train_ds)} 条训练数据")
            
            if len(train_ds) == 0:
                logger.error("❌ 没有训练数据！请检查数据文件")
                return
                
        except Exception as e:
            logger.error(f"❌ 数据准备失败: {e}")
            return
        
        # 3. 保存配置
        output_dir = self.config['data']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        self.save_config(output_dir)
        
        # 4. 开始训练
        logger.info("🎯 开始模型训练...")
        try:
            # 基础训练参数
            fit_args = {
                'train_ds': train_ds,
                'output_dir': output_dir,
                'batch_size': self.config['training']['batch_size'],
                'gradient_accumulation_steps': self.config['training']['gradient_accumulation_steps'],
                'epochs': self.config['training']['num_epochs'],
                'learning_rate': self.config['training']['learning_rate'],
                'save_steps': self.config['training']['save_steps'],
                'warmup_steps': self.config['training']['warmup_steps'],
                'loss_kwargs': self.config['loss'],
                'logging_steps': self.config['training']['logging_steps'],
            }
            
            # FP16配置
            if self.config['training']['fp16'] and torch.cuda.is_available():
                fit_args['fp16'] = True
                logger.info("🔧 启用FP16混合精度训练")
            else:
                fit_args['fp16'] = False
                logger.info("🔧 禁用FP16")
            
            # 开始训练
            angle.fit(**fit_args)
            
            logger.info(f"🎉 训练完成！模型已保存至: {output_dir}")
            
        except torch.cuda.OutOfMemoryError:
            logger.error("💥 GPU内存不足！")
            self.handle_oom_error()
            
        except Exception as e:
            logger.error(f"💥 训练过程中出现错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 尝试不使用FP16
            if self.config['training']['fp16']:
                logger.info("🔄 尝试禁用FP16...")
                try:
                    fit_args['fp16'] = False
                    angle.fit(**fit_args)
                    logger.info(f"🎉 训练完成！模型已保存至: {output_dir}")
                except Exception as e2:
                    logger.error(f"💥 重试也失败了: {e2}")
                    raise
            else:
                raise
    
    def handle_oom_error(self):
        """处理内存不足错误"""
        logger.info("💡 内存优化建议:")
        logger.info("  1. 降低 batch_size")
        logger.info("  2. 降低 max_length")
        logger.info("  3. 禁用 FP16")
        logger.info("  4. 使用更小的模型")


def main():
    """主函数"""
    try:
        # 可以指定不同的配置文件
        config_path = r"train\v2_another\config\train_config.json"
        
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            # 尝试使用绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "config", "train_config.json")
            if not os.path.exists(config_path):
                # 尝试在当前目录查找
                config_path = "train_config.json"
                if not os.path.exists(config_path):
                    # 列出可能的配置文件
                    logger.error("找不到配置文件，请检查以下位置:")
                    for dirpath, dirnames, filenames in os.walk(current_dir):
                        for filename in filenames:
                            if 'config' in filename and filename.endswith('.json'):
                                logger.info(f"  {os.path.join(dirpath, filename)}")
                    raise FileNotFoundError(f"找不到配置文件")
        
        logger.info(f"📋 使用配置文件: {config_path}")
        
        trainer = AngleTrainerWrapper(config_path)
        trainer.train()
    except KeyboardInterrupt:
        logger.info("⏹️ 训练被用户中断")
    except Exception as e:
        logger.error(f"💥 程序执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()