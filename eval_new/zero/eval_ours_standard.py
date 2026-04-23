"""
Zero-shot 评估脚本
- 不训练任何分类器
- 直接使用预训练模型的 [CLS] 特征 + 固定相似度/距离分类
- 输出混淆矩阵、分类报告等详细信息
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    precision_score, recall_score
)
from transformers import AutoModel, AutoTokenizer

# ==================== 配置 ====================
# 模型路径（使用基础预训练模型，不加任何 projection）
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"

# 数据路径
TRAIN_CSV = r"asap-master/data/train.csv"
TEST_CSV = r"asap-master/data/test.csv"

# 结果保存目录
OUTPUT_DIR = r"eval_new/zero/result/zero_shot_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aspect 映射（英文名 -> 中文名）
ASPECT_MAPPING = {
    'Location#Transportation': '交通',
    'Location#Downtown': '地段',
    'Location#Easy_to_find': '位置易找度',
    'Service#Queue': '排队',
    'Service#Hospitality': '服务态度',
    'Service#Parking': '停车',
    'Service#Timely': '上菜速度',
    'Price#Level': '价格水平',
    'Price#Cost_effective': '性价比',
    'Price#Discount': '折扣优惠',
    'Ambience#Decoration': '装修装饰',
    'Ambience#Noise': '噪音环境',
    'Ambience#Space': '空间',
    'Ambience#Sanitary': '卫生',
    'Food#Portion': '份量',
    'Food#Taste': '口味',
    'Food#Appearance': '外观卖相',
    'Food#Recommend': '推荐度'
}

# 超参数
BATCH_SIZE = 128
MAX_LEN = 512
RANDOM_STATE = 42

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 提示模板（Zero-shot 不需要 prompt）====================
# Zero-shot 直接使用原始文本，不添加任何 aspect prompt

def clean_review(text):
    """清洗文本"""
    import re
    text = str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\\', '', text)
    text = text.replace('//', ' ')
    return text.strip()

# ==================== 模型定义 ====================
class ZeroShotModel(nn.Module):
    """简单的预训练模型封装，用于提取 [CLS] 特征"""
    def __init__(self, model_name_or_path, max_length=512):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # 取 CLS 向量 (第一个 token)
        pooled = outputs.last_hidden_state[:, 0, :]
        return pooled

    def to(self, device):
        self.model = self.model.to(device)
        return self

def load_zero_shot_model(model_name_or_path, max_length=512):
    """加载 Zero-shot 模型"""
    model = ZeroShotModel(model_name_or_path, max_length=max_length)
    return model

def encode_texts(model, texts, batch_size=BATCH_SIZE, normalize=False):
    """批量编码文本，返回 numpy 数组"""
    model.eval()
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = model.tokenizer(batch, padding='longest', truncation=True,
                                     max_length=model.max_length, return_tensors='pt')
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            emb = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            if normalize:
                emb = F.normalize(emb, p=2, dim=-1)
            all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)

def load_aspect_data(csv_path, aspect_en):
    """加载并过滤某个 aspect 的数据，只保留正负样本"""
    df = pd.read_csv(csv_path)
    df['review'] = df['review'].apply(clean_review)
    # 只保留 1 和 -1
    df = df[df[aspect_en].isin([1, -1])].copy()
    df['label'] = df[aspect_en].astype(int)
    return df

# ==================== Zero-shot 分类函数 ====================
def zero_shot_classify_by_prototype(train_embeddings, train_labels, test_embeddings):
    """
    Zero-shot 分类：使用训练集构建类别原型（质心），然后对测试集进行分类
    
    方法：
    1. 计算每个类别的原型向量（所有训练样本的平均 embedding）
    2. 对于每个测试样本，计算其与两个类别原型的余弦相似度
    3. 选择相似度更高的类别作为预测
    
    这比随机初始化要好，因为它利用了训练集的信息来构建类别边界。
    真正的 Zero-shot 应该使用预定义的类别描述 embedding，但这里我们使用训练集原型。
    
    注意：严格意义上的 Zero-shot 不应该使用训练集标签来构建原型。
    但 ABSA 任务中，类别是预定义的（positive/negative），
    我们可以使用类别描述文本（如 "positive sentiment" 和 "negative sentiment"）的 embedding 作为原型。
    
    这里提供两种模式：
    - mode='train_prototype': 使用训练集构建原型（弱 zero-shot）
    - mode='description_prototype': 使用类别描述文本构建原型（强 zero-shot）
    """
    # 计算正负类原型（训练集均值）
    pos_prototype = train_embeddings[train_labels == 1].mean(axis=0)
    neg_prototype = train_embeddings[train_labels == -1].mean(axis=0)
    
    # 归一化原型
    pos_prototype = pos_prototype / (np.linalg.norm(pos_prototype) + 1e-8)
    neg_prototype = neg_prototype / (np.linalg.norm(neg_prototype) + 1e-8)
    
    # 归一化测试 embedding
    test_embeddings_norm = test_embeddings / (np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # 计算相似度
    sim_pos = test_embeddings_norm @ pos_prototype
    sim_neg = test_embeddings_norm @ neg_prototype
    
    # 预测：相似度高的为正类
    y_pred = np.where(sim_pos > sim_neg, 1, -1)
    
    return y_pred, sim_pos, sim_neg

def zero_shot_classify_by_description(model, texts, aspect_cn):
    """
    严格的 Zero-shot 分类：使用类别描述文本的 embedding 作为原型
    完全不使用训练集标签
    
    类别描述：
    - Positive: f"关于{aspect_cn}的正面评价"
    - Negative: f"关于{aspect_cn}的负面评价"
    """
    # 构建类别描述
    pos_desc = f"关于{aspect_cn}的正面评价"
    neg_desc = f"关于{aspect_cn}的负面评价"
    
    # 编码描述
    pos_emb = encode_texts(model, [pos_desc], batch_size=1, normalize=True)[0]
    neg_emb = encode_texts(model, [neg_desc], batch_size=1, normalize=True)[0]
    
    # 编码测试文本
    test_embeddings = encode_texts(model, texts, normalize=True)
    
    # 计算相似度
    sim_pos = test_embeddings @ pos_emb
    sim_neg = test_embeddings @ neg_emb
    
    # 预测
    y_pred = np.where(sim_pos > sim_neg, 1, -1)
    
    return y_pred, sim_pos, sim_neg

# ==================== 核心评估函数 ====================
def evaluate_zero_shot(model, train_df, test_df, aspect_cn, mode='train_prototype'):
    """
    Zero-shot 评估
    
    Args:
        model: 预训练模型
        train_df: 训练集（仅用于构建原型，不用于训练分类器）
        test_df: 测试集
        aspect_cn: 方面中文名
        mode: 'train_prototype' 或 'description_prototype'
    """
    # Zero-shot: 不使用 prompt，直接使用原始文本
    train_texts = [row['review'] for _, row in train_df.iterrows()]
    train_labels = train_df['label'].values
    test_texts = [row['review'] for _, row in test_df.iterrows()]
    test_labels = test_df['label'].values

    if len(np.unique(train_labels)) < 2:
        print(f"  跳过 {aspect_cn}: 训练集中只有一个类别")
        return None

    print(f"  编码训练集 ({len(train_texts)} 条)...")
    train_embeddings = encode_texts(model, train_texts, batch_size=BATCH_SIZE, normalize=False)
    print(f"  编码测试集 ({len(test_texts)} 条)...")
    test_embeddings = encode_texts(model, test_texts, batch_size=BATCH_SIZE, normalize=False)

    if mode == 'train_prototype':
        y_pred, sim_pos, sim_neg = zero_shot_classify_by_prototype(train_embeddings, train_labels, test_embeddings)
    else:  # description_prototype
        y_pred, sim_pos, sim_neg = zero_shot_classify_by_description(model, test_texts, aspect_cn)

    # 计算指标
    acc = accuracy_score(test_labels, y_pred)
    f1_macro = f1_score(test_labels, y_pred, average='macro')
    f1_pos = f1_score(test_labels, y_pred, pos_label=1, zero_division=0)
    f1_neg = f1_score(test_labels, y_pred, pos_label=-1, zero_division=0)
    prec_pos = precision_score(test_labels, y_pred, pos_label=1, zero_division=0)
    rec_pos = recall_score(test_labels, y_pred, pos_label=1, zero_division=0)
    prec_neg = precision_score(test_labels, y_pred, pos_label=-1, zero_division=0)
    rec_neg = recall_score(test_labels, y_pred, pos_label=-1, zero_division=0)

    cm = confusion_matrix(test_labels, y_pred, labels=[-1, 1])
    report = classification_report(test_labels, y_pred, labels=[-1, 1],
                                   target_names=['negative', 'positive'],
                                   output_dict=True)

    result = {
        'aspect': aspect_cn,
        'mode': mode,
        'train_samples': len(train_labels),
        'train_pos': int(sum(train_labels == 1)),
        'train_neg': int(sum(train_labels == -1)),
        'test_samples': len(test_labels),
        'test_pos': int(sum(test_labels == 1)),
        'test_neg': int(sum(test_labels == -1)),
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_positive': f1_pos,
        'f1_negative': f1_neg,
        'precision_positive': prec_pos,
        'recall_positive': rec_pos,
        'precision_negative': prec_neg,
        'recall_negative': rec_neg,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

    print(f"\n  [{aspect_cn}] 测试样本数: {len(test_labels)}")
    print(f"    Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f}")
    print(f"    Positive: Prec={prec_pos:.4f} Rec={rec_pos:.4f} F1={f1_pos:.4f}")
    print(f"    Negative: Prec={prec_neg:.4f} Rec={rec_neg:.4f} F1={f1_neg:.4f}")
    print(f"    Confusion Matrix (rows: true, cols: pred) [-1, 1]:")
    print(f"      {cm[0]}  (true negative)")
    print(f"      {cm[1]}  (true positive)")

    return result


# ==================== 主函数 ====================
def main():
    # 配置
    MODE = 'train_prototype'  # 可选: 'train_prototype' 或 'description_prototype'
    
    print("="*70)
    print("Zero-shot 评估 (使用训练集原型/类别描述进行分类)")
    print(f"模型: {MODEL_NAME}")
    print(f"模式: {MODE}")
    print(f"设备: {DEVICE}")
    print("="*70)

    # 加载模型
    try:
        print(f"正在加载模型...")
        model = load_zero_shot_model(MODEL_NAME, max_length=MAX_LEN)
        model = model.to(DEVICE)
        model.eval()
        print(f"✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print(f"\n提示：")
        print(f"  1. 如果使用 Hugging Face 模型名，请确保网络连接正常")
        print(f"  2. 如果网络问题，可以设置镜像: set HF_ENDPOINT=https://hf-mirror.com")
        print(f"  3. 或下载模型到本地，然后修改 MODEL_NAME 为本地路径")
        return

    # 加载完整训练集和测试集
    try:
        train_full = pd.read_csv(TRAIN_CSV)
        test_full = pd.read_csv(TEST_CSV)
        print(f"✅ 数据加载成功")
        print(f"训练集原始行数: {len(train_full)}，测试集: {len(test_full)}")
    except FileNotFoundError as e:
        print(f"❌ 数据文件不存在: {e}")
        print(f"请检查路径: {TRAIN_CSV} 和 {TEST_CSV}")
        return

    all_results = {}
    total_aspects = len(ASPECT_MAPPING)
    processed_aspects = 0
    
    for aspect_en, aspect_cn in ASPECT_MAPPING.items():
        processed_aspects += 1
        print(f"\n{'='*50}")
        print(f"处理 Aspect [{processed_aspects}/{total_aspects}]: {aspect_en} ({aspect_cn})")
        print(f"{'='*50}")
        
        try:
            train_df = load_aspect_data(TRAIN_CSV, aspect_en)
            test_df = load_aspect_data(TEST_CSV, aspect_en)

            if len(train_df) == 0 or len(test_df) == 0:
                print(f"  ⚠️ 跳过: 训练集或测试集无有效数据 (train={len(train_df)}, test={len(test_df)})")
                continue

            print(f"  训练集有效样本: {len(train_df)} (正样本: {sum(train_df['label']==1)}, 负样本: {sum(train_df['label']==-1)})")
            print(f"  测试集有效样本: {len(test_df)} (正样本: {sum(test_df['label']==1)}, 负样本: {sum(test_df['label']==-1)})")
            
            res = evaluate_zero_shot(model, train_df, test_df, aspect_cn, mode=MODE)
            if res:
                all_results[aspect_cn] = res
        except Exception as e:
            print(f"  ❌ 处理 {aspect_cn} 时出错: {e}")
            continue

    # 计算总体平均
    if all_results:
        avg_acc = np.mean([r['accuracy'] for r in all_results.values()])
        avg_f1_macro = np.mean([r['f1_macro'] for r in all_results.values()])
        avg_f1_pos = np.mean([r['f1_positive'] for r in all_results.values()])
        avg_f1_neg = np.mean([r['f1_negative'] for r in all_results.values()])
        
        print("\n" + "="*70)
        print("评估完成 - 总体结果")
        print("="*70)
        print(f"成功评估的 Aspect 数量: {len(all_results)}/{total_aspects}")
        print(f"总体平均 Accuracy:    {avg_acc:.4f}")
        print(f"总体平均 Macro F1:    {avg_f1_macro:.4f}")
        print(f"总体平均 Positive F1: {avg_f1_pos:.4f}")
        print(f"总体平均 Negative F1: {avg_f1_neg:.4f}")
        print("="*70)

        # 保存 JSON 结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(OUTPUT_DIR, f"zero_shot_results_{MODE}_{timestamp}.json")
        
        output_data = {
            "summary": {
                "model": MODEL_NAME,
                "mode": MODE,
                "device": str(DEVICE),
                "timestamp": timestamp,
                "total_aspects": total_aspects,
                "evaluated_aspects": len(all_results),
                "average_accuracy": avg_acc,
                "average_f1_macro": avg_f1_macro,
                "average_f1_positive": avg_f1_pos,
                "average_f1_negative": avg_f1_neg
            },
            "details": all_results
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ JSON 结果已保存至: {json_path}")

        # 保存文本报告
        txt_path = os.path.join(OUTPUT_DIR, f"zero_shot_report_{MODE}_{timestamp}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"Zero-shot 评估报告\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型: {MODEL_NAME}\n")
            f.write(f"模式: {MODE}\n")
            f.write(f"设备: {DEVICE}\n")
            f.write("="*70 + "\n")
            f.write(f"\n总体统计:\n")
            f.write(f"  评估 Aspect 数量: {len(all_results)}/{total_aspects}\n")
            f.write(f"  平均 Accuracy: {avg_acc:.4f}\n")
            f.write(f"  平均 Macro F1: {avg_f1_macro:.4f}\n")
            f.write(f"  平均 Positive F1: {avg_f1_pos:.4f}\n")
            f.write(f"  平均 Negative F1: {avg_f1_neg:.4f}\n")
            f.write("\n" + "="*70 + "\n")
            f.write("\n详细结果:\n")
            f.write("-"*70 + "\n")
            
            for aspect, res in all_results.items():
                f.write(f"\n【{aspect}】\n")
                f.write(f"  训练样本数: {res['train_samples']}\n")
                f.write(f"  测试样本数: {res['test_samples']}\n")
                f.write(f"  Accuracy: {res['accuracy']:.4f}\n")
                f.write(f"  Macro F1: {res['f1_macro']:.4f}\n")
                f.write(f"  Positive F1: {res['f1_positive']:.4f}\n")
                f.write(f"  Negative F1: {res['f1_negative']:.4f}\n")
                cm = res['confusion_matrix']
                f.write(f"  混淆矩阵 (真实\\预测):\n")
                f.write(f"    TN={cm[0][0]}, FP={cm[0][1]}\n")
                f.write(f"    FN={cm[1][0]}, TP={cm[1][1]}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("\n各指标统计:\n")
            f.write("-"*70 + "\n")
            
            metrics = ['accuracy', 'f1_macro', 'f1_positive', 'f1_negative']
            for metric in metrics:
                values = [r[metric] for r in all_results.values()]
                f.write(f"\n{metric}:\n")
                f.write(f"  Mean: {np.mean(values):.4f}\n")
                f.write(f"  Std:  {np.std(values):.4f}\n")
                f.write(f"  Min:  {np.min(values):.4f}\n")
                f.write(f"  Max:  {np.max(values):.4f}\n")

        print(f"✅ 文本报告已保存至: {txt_path}")
        
        # 显示每个 Aspect 的简要结果
        print("\n" + "="*70)
        print("各 Aspect 结果汇总:")
        print("-"*70)
        print(f"{'Aspect':<20} {'Acc':<8} {'F1_macro':<10} {'F1_pos':<8} {'F1_neg':<8} {'样本数':<8}")
        print("-"*70)
        for aspect, res in all_results.items():
            print(f"{aspect:<20} {res['accuracy']:.4f}   {res['f1_macro']:.4f}    "
                  f"{res['f1_positive']:.4f}   {res['f1_negative']:.4f}   {res['test_samples']:<8}")
        print("="*70)
        
    else:
        print("\n❌ 没有有效的评估结果")

if __name__ == "__main__":
    main()