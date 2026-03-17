import torch
import os
import sys
import gc
import json
import logging

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

from angle_emb import AnglE
from data_converter import TripletDataConverter

class AngleTrainer:
    def __init__(self, config_path="config/train_config.json"):
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
        """加载和准备数据"""
        try:
            input_file = self.config['data']['input_jsonl_file']
            logger.info(f"📥 正在加载数据集: {input_file}")
            
            # 使用数据转换器加载和转换数据
            dataset = TripletDataConverter.load_and_convert(input_file)
            
            # 显示样本示例
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
                self.config['model']['name'],
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
            return
        
        # 2. 加载数据
        try:
            train_ds = self.load_and_prepare_data()
        except Exception as e:
            logger.error(f"❌ 数据准备失败: {e}")
            return
        
        # 3. 保存配置
        output_dir = self.config['data']['output_dir']
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
            
            angle.fit(**fit_args)
            
            logger.info(f"🎉 训练完成！模型已保存至: {output_dir}")
            
        except torch.cuda.OutOfMemoryError:
            logger.error("💥 GPU内存不足！")
            self.handle_oom_error()
            
        except Exception as e:
            logger.error(f"💥 训练过程中出现错误: {e}")
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
        config_path = r"train\config\train_config.json"
        trainer = AngleTrainer(config_path)
        trainer.train()
    except KeyboardInterrupt:
        logger.info("⏹️ 训练被用户中断")
    except Exception as e:
        logger.error(f"💥 程序执行失败: {e}")
        raise

if __name__ == '__main__':
    main()