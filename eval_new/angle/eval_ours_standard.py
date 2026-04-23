"""
Ours (AnglE + ZRCP) 标准评估脚本
- 使用训练集训练逻辑回归分类器
- 在测试集上评估
- 输出混淆矩阵、分类报告等详细信息
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    precision_score, recall_score
)
from transformers import AutoModel, AutoTokenizer

# ==================== 配置 ====================
# 模型路径（训练好的 checkpoint，包含 complex_proj.bin）
MODEL_PATH = r"checkpoints_standard_angle/v1"   # 请修改为您的实际路径

# 数据路径
TRAIN_CSV = r"asap-master/data/train.csv"
TEST_CSV  = r"asap-master/data/test.csv"

# 结果保存目录
OUTPUT_DIR = r"eval_new\angle\result\eval_ours_standard_results"
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

# ==================== 模型定义（与训练时一致）====================
class ComplexProjection(nn.Module):
    """残差可学习复数空间投影层 (Residual AP-Proj)"""
    def __init__(self, hidden_size):
        super().__init__()
        self.half_size = hidden_size // 2
        self.proj_re = nn.Linear(self.half_size, self.half_size)
        self.proj_im = nn.Linear(self.half_size, self.half_size)
        nn.init.zeros_(self.proj_re.weight)
        nn.init.zeros_(self.proj_im.weight)
        nn.init.zeros_(self.proj_re.bias)
        nn.init.zeros_(self.proj_im.bias)

    def forward(self, x):
        re_orig, im_orig = torch.chunk(x, 2, dim=1)
        re = re_orig + self.proj_re(re_orig)
        im = im_orig + self.proj_im(im_orig)
        return torch.cat([re, im], dim=1)

class AnglE_Eval(nn.Module):
    def __init__(self, model_name_or_path, max_length=512):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_length = max_length

        hidden_size = self.model.config.hidden_size
        self.model.add_module('complex_proj', ComplexProjection(hidden_size))
        proj_path = os.path.join(model_name_or_path, 'complex_proj.bin')
        if os.path.exists(proj_path):
            self.model.complex_proj.load_state_dict(torch.load(proj_path, map_location='cpu'))
            print(f"✅ 加载投影层权重: {proj_path}")
        else:
            print("⚠️ 未找到 complex_proj.bin，将不使用投影层")
            delattr(self.model, 'complex_proj')

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS
        if hasattr(self.model, 'complex_proj'):
            pooled = self.model.complex_proj(pooled)
        return pooled

    def to(self, device):
        self.model = self.model.to(device)
        return self

def load_angle_model(model_path, max_length=512):
    model = AnglE_Eval(model_path, max_length=max_length)
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
                emb = nn.functional.normalize(emb, p=2, dim=-1)
            all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)

def create_ap_text(aspect_cn, review):
    """构造 Aspect-Polarity 输入文本"""
    return f"Aspect: {aspect_cn}, Context: {review}"

def clean_review(text):
    import re
    text = str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\\', '', text)
    text = text.replace('//', ' ')
    return text.strip()

def load_aspect_data(csv_path, aspect_en):
    """加载并过滤某个 aspect 的数据，只保留正负样本"""
    df = pd.read_csv(csv_path)
    df['review'] = df['review'].apply(clean_review)
    # 只保留 1 和 -1
    df = df[df[aspect_en].isin([1, -1])].copy()
    df['label'] = df[aspect_en].astype(int)
    return df

# ==================== 核心评估函数 ====================
def evaluate_aspect_standard(model, train_df, test_df, aspect_cn):
    """
    使用训练集嵌入训练分类器，在测试集上评估
    返回详细指标字典
    """
    # 构造文本
    train_texts = [create_ap_text(aspect_cn, row['review']) for _, row in train_df.iterrows()]
    train_labels = train_df['label'].values
    test_texts = [create_ap_text(aspect_cn, row['review']) for _, row in test_df.iterrows()]
    test_labels = test_df['label'].values

    if len(np.unique(train_labels)) < 2:
        print(f"  跳过 {aspect_cn}: 训练集中只有一个类别")
        return None

    print(f"  编码训练集 ({len(train_texts)} 条)...")
    X_train = encode_texts(model, train_texts, batch_size=BATCH_SIZE, normalize=False)
    print(f"  编码测试集 ({len(test_texts)} 条)...")
    X_test = encode_texts(model, test_texts, batch_size=BATCH_SIZE, normalize=False)

    # 训练逻辑回归（平衡权重）
    clf = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE)
    clf.fit(X_train, train_labels)
    y_pred = clf.predict(X_test)

    # 计算指标
    acc = accuracy_score(test_labels, y_pred)
    f1_macro = f1_score(test_labels, y_pred, average='macro')
    f1_pos = f1_score(test_labels, y_pred, pos_label=1)
    f1_neg = f1_score(test_labels, y_pred, pos_label=-1)
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
        'train_samples': len(train_labels),
        'test_samples': len(test_labels),
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
    print("="*70)
    print("Ours (AnglE+ZRCP) 标准评估 (训练集训练LR，测试集评估)")
    print(f"模型: {MODEL_PATH}")
    print(f"设备: {DEVICE}")
    print("="*70)

    # 检查模型路径
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型路径不存在: {MODEL_PATH}")
        return

    # 加载模型
    model = load_angle_model(MODEL_PATH, max_length=MAX_LEN)
    model = model.to(DEVICE)
    model.eval()

    # 加载完整训练集和测试集
    train_full = pd.read_csv(TRAIN_CSV)
    test_full = pd.read_csv(TEST_CSV)
    print(f"训练集原始行数: {len(train_full)}，测试集: {len(test_full)}")

    all_results = {}
    for aspect_en, aspect_cn in ASPECT_MAPPING.items():
        print(f"\n{'='*50}")
        print(f"处理 Aspect: {aspect_en} ({aspect_cn})")
        train_df = load_aspect_data(TRAIN_CSV, aspect_en)
        test_df = load_aspect_data(TEST_CSV, aspect_en)

        if len(train_df) == 0 or len(test_df) == 0:
            print(f"  跳过: 训练集或测试集无有效数据 (train={len(train_df)}, test={len(test_df)})")
            continue

        res = evaluate_aspect_standard(model, train_df, test_df, aspect_cn)
        if res:
            all_results[aspect_cn] = res

    # 计算总体平均
    if all_results:
        avg_acc = np.mean([r['accuracy'] for r in all_results.values()])
        avg_f1_macro = np.mean([r['f1_macro'] for r in all_results.values()])
        print("\n" + "="*70)
        print(f"总体平均 Accuracy: {avg_acc:.4f}")
        print(f"总体平均 Macro F1: {avg_f1_macro:.4f}")
        print("="*70)

        # 保存 JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(OUTPUT_DIR, f"ours_results_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # 保存文本报告
        txt_path = os.path.join(OUTPUT_DIR, f"ours_report_{timestamp}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"Ours (AnglE+ZRCP) 标准评估报告\n")
            f.write(f"时间: {datetime.now()}\n")
            f.write(f"模型: {MODEL_PATH}\n")
            f.write(f"平均 Accuracy: {avg_acc:.4f}\n")
            f.write(f"平均 Macro F1: {avg_f1_macro:.4f}\n")
            f.write("="*70 + "\n\n")
            for aspect, res in all_results.items():
                f.write(f"\n--- {aspect} ---\n")
                f.write(f"训练样本数: {res['train_samples']}, 测试样本数: {res['test_samples']}\n")
                f.write(f"Accuracy: {res['accuracy']:.4f}\n")
                f.write(f"Macro F1: {res['f1_macro']:.4f}\n")
                f.write(f"Positive: Prec={res['precision_positive']:.4f} Rec={res['recall_positive']:.4f} F1={res['f1_positive']:.4f}\n")
                f.write(f"Negative: Prec={res['precision_negative']:.4f} Rec={res['recall_negative']:.4f} F1={res['f1_negative']:.4f}\n")
                f.write("混淆矩阵 (真实\\预测):\n")
                cm = res['confusion_matrix']
                f.write(f"  negative: {cm[0]}\n")
                f.write(f"  positive: {cm[1]}\n\n")

        print(f"\n✅ 结果已保存至:\n  {json_path}\n  {txt_path}")
    else:
        print("❌ 没有有效的评估结果")

if __name__ == "__main__":
    main()