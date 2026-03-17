# angle_diagnostic_simple.py
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from transformers import AutoModel, AutoTokenizer
from typing import Dict

# 添加AnglE库到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = r'D:\WorkSpace\AnglE_yj'
angle_root = os.path.join(project_root, 'AnglE')

if os.path.exists(angle_root) and angle_root not in sys.path:
    sys.path.insert(0, angle_root)
    print(f"✅ 已添加AnglE根目录到路径: {angle_root}")

# ============================================
# 诊断函数
# ============================================

def test_our_implementation():
    """测试我们的实现"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 测试文本
    test_texts = [
        "Aspect: 口味, Context: 这家餐厅的菜很好吃，味道正宗",
        "Aspect: 口味, Context: 菜太难吃了，完全没有味道",
    ]
    
    # 模型路径
    model_path = r'checkpoints\angle_alce_encoder_final_V2\checkpoint-655'
    
    print(f"\n测试模型: {model_path}")
    print(f"测试文本数量: {len(test_texts)}")
    
    # 1. 直接使用transformers提取（基线）
    print("\n" + "="*80)
    print("1. 直接使用transformers提取嵌入（基线）")
    print("="*80)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    
    # 准备输入
    inputs = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=192,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 直接取CLS向量
    baseline_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    print(f"基线嵌入形状: {baseline_embeddings.shape}")
    print(f"基线嵌入均值: {np.mean(baseline_embeddings):.6f}")
    print(f"基线嵌入标准差: {np.std(baseline_embeddings):.6f}")
    print(f"基线嵌入范数: {np.linalg.norm(baseline_embeddings, axis=1)}")
    
    # 2. 使用不同的池化策略
    print("\n" + "="*80)
    print("2. 测试不同池化策略")
    print("="*80)
    
    def get_pooling(outputs, inputs, pooling_strategy, padding_side):
        """池化函数"""
        if pooling_strategy == 'cls':
            return outputs[:, 0]
        elif pooling_strategy == 'mean':
            input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(outputs.size()).float()
            sum_embeddings = torch.sum(outputs * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        elif pooling_strategy == 'max':
            input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(outputs.size()).float()
            outputs[input_mask_expanded == 0] = -1e9
            return torch.max(outputs, 1)[0]
        elif pooling_strategy == 'last':
            batch_size = inputs['input_ids'].shape[0]
            sequence_lengths = -1 if padding_side == 'left' else inputs['attention_mask'].sum(dim=1) - 1
            return outputs[torch.arange(batch_size, device=outputs.device), sequence_lengths]
        else:
            return outputs[:, 0]
    
    for strategy in ['cls', 'mean', 'max', 'last']:
        pooled = get_pooling(outputs.last_hidden_state, inputs, strategy, tokenizer.padding_side)
        pooled_np = pooled.cpu().numpy()
        
        print(f"\n策略 '{strategy}':")
        print(f"  形状: {pooled_np.shape}")
        print(f"  均值: {np.mean(pooled_np):.6f}")
        print(f"  范数: {np.linalg.norm(pooled_np, axis=1)}")
    
    # 3. 检查模型配置
    print("\n" + "="*80)
    print("3. 检查模型配置")
    print("="*80)
    
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    
    print(f"模型类型: {config.model_type}")
    print(f"架构: {config.architectures}")
    print(f"隐藏层大小: {config.hidden_size}")
    print(f"层数: {config.num_hidden_layers}")
    print(f"最大位置编码: {config.max_position_embeddings}")
    
    # 4. 检查是否有特殊处理
    print("\n" + "="*80)
    print("4. 检查模型文件")
    print("="*80)
    
    # 检查配置文件
    angle_config_path = os.path.join(model_path, 'angle_config.json')
    if os.path.exists(angle_config_path):
        import json
        with open(angle_config_path, 'r', encoding='utf-8') as f:
            angle_config = json.load(f)
        print("✅ 找到angle_config.json:")
        for key, value in angle_config.items():
            print(f"  {key}: {value}")
    else:
        print("❌ 未找到angle_config.json")
    
    # 5. 创建一个简单的测试评估
    print("\n" + "="*80)
    print("5. 简单分类测试")
    print("="*80)
    
    # 模拟评估：用同一个模型提取特征，看是否可分
    from sklearn.linear_model import LogisticRegression
    
    # 创建简单测试数据
    positive_texts = [
        "Aspect: 口味, Context: 味道很好",
        "Aspect: 口味, Context: 非常好吃",
        "Aspect: 口味, Context: 美味可口",
        "Aspect: 口味, Context: 味道不错",
    ]
    
    negative_texts = [
        "Aspect: 口味, Context: 很难吃",
        "Aspect: 口味, Context: 味道很差",
        "Aspect: 口味, Context: 不好吃",
        "Aspect: 口味, Context: 味道一般",
    ]
    
    all_texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [-1] * len(negative_texts)
    
    # 提取嵌入
    test_inputs = tokenizer(
        all_texts,
        padding=True,
        truncation=True,
        max_length=192,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        test_outputs = model(**test_inputs)
    
    # 使用CLS池化
    embeddings = test_outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    # 训练简单分类器
    clf = LogisticRegression(random_state=42)
    clf.fit(embeddings, labels)
    
    accuracy = clf.score(embeddings, labels)
    print(f"分类准确率: {accuracy:.4f}")
    print(f"说明嵌入向量{'具有' if accuracy > 0.8 else '不具有'}良好的线性可分性")


if __name__ == '__main__':
    test_our_implementation()