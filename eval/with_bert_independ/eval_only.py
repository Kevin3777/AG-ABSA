import pandas as pd
import numpy as np
import torch
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import logging

# --- 🛠️ 路径自动加固 (必须放在所有 import 之前) ---
def setup_path():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设脚本在 D:\WorkSpace\AnglE_yj\rag_build\，需要向上两级找到 AnglE
    project_root = r'D:\WorkSpace\AnglE_yj' 
    angle_root = os.path.join(project_root, 'AnglE')
    
    if os.path.exists(angle_root) and angle_root not in sys.path:
        sys.path.insert(0, angle_root)
        print(f"✅ 已添加 AnglE 根目录到路径: {angle_root}")
    elif angle_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"✅ 已添加项目根目录到路径: {project_root}")

setup_path()

from angle_emb import AnglE


# --- 配置参数 (与您的环境保持一致) ---
TRAIN_CSV_FILE = r'asap-master\data\train.csv'
TEST_CSV_FILE = r'asap-master\data\test.csv'

# ⚠️ 修复点：模型路径指向您最终成功的 Checkpoint 目录
MODEL_PATH = r'checkpoint_dotproduct\dot_product_model\best_model' 
BASE_MODEL_NAME = "hfl/chinese-roberta-wwm-ext" # 对比基线模型

# 评估 Aspect
ASPECT_TO_EVALUATE = 'Food#Taste'
ASPECT_MAPPING = {'Food#Taste': '口味'}
BATCH_SIZE = 32

# --- 核心评估函数 ---
def clean_review_text(text):
    import re
    text = str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\\', '', text) 
    text = text.replace('//', ' ')    
    return text.strip()

def load_data_and_filter(file_path):
    df = pd.read_csv(file_path)
    df['review'] = df['review'].apply(clean_review_text)
    # 只保留有明确情感（1 或 -1）的样本
    df = df[df[ASPECT_TO_EVALUATE].isin([1, -1])].copy()
    df['label'] = df[ASPECT_TO_EVALUATE].astype(int)
    return df

def create_ap_pair(aspect_chinese_name, review_context):
    """创建 AnglE 所需的 AP Pair 输入格式"""
    return f"Aspect: {aspect_chinese_name}, Context: {review_context}"

def get_embeddings(model_path, df, aspect_name_cn, is_angle_model=True, batch_size=32):
    """提取 AnglE/BERT 嵌入向量"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 加载模型
    if is_angle_model:
        # AnglE 模型：使用其内置的加载逻辑
        model = AnglE.from_pretrained(model_path, pooling_strategy='cls').to(device)
    else:
        # 基线 BERT 模型：需要手动加载 Tokenizer 和 Model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device)

    # 2. 准备输入文本
    if is_angle_model:
        texts = [create_ap_pair(aspect_name_cn, text) for text in df['review'].tolist()]
    else:
        texts = df['review'].tolist()

    # 3. 提取向量
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting {'AnglE' if is_angle_model else 'BERT'} Embeddings"):
        batch_texts = texts[i:i + batch_size]
        
        if is_angle_model:
            # AnglE encode (无需更改)
            batch_embeddings = model.encode(batch_texts, to_numpy=True)
            embeddings.append(batch_embeddings)
        else:
            # 🌟 关键修复：强制设置 max_length=512 和 truncation=True
            inputs = tokenizer(
                batch_texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, # 强制截断过长的序列
                max_length=512 # BERT 的最大长度
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # CLS 向量提取
            embeddings.append(outputs[0][:, 0, :].cpu().numpy())

    return np.concatenate(embeddings, axis=0)

def evaluate_feature_quality():
    print("--- 1. 数据准备 ---")
    df_train = load_data_and_filter(TRAIN_CSV_FILE)
    df_test = load_data_and_filter(TEST_CSV_FILE)
    
    aspect_cn = ASPECT_MAPPING[ASPECT_TO_EVALUATE]
    
    # --- 2. 特征提取 ---
    
    # 2.1 您的方案：AnglE-ALCE 向量
    print("\n--- 2.1 提取 AnglE-ALCE 训练向量 (SOTA 特征) ---")
    X_train_angle = get_embeddings(MODEL_PATH, df_train, aspect_cn, is_angle_model=True)
    X_test_angle = get_embeddings(MODEL_PATH, df_test, aspect_cn, is_angle_model=True)
    
    # 2.2 基线方案：原始 BERT CLS 向量（注释掉这三行）
    # print("\n--- 2.2 提取 BERT-CLS 向量 (基线特征) ---")
    # X_train_bert = get_embeddings(BASE_MODEL_NAME, df_train, aspect_cn, is_angle_model=False)
    # X_test_bert = get_embeddings(BASE_MODEL_NAME, df_test, aspect_cn, is_angle_model=False)

    Y_train = df_train['label'].values
    Y_test = df_test['label'].values

    # --- 3. 训练 Logistic Regression 分类器并对比 ---
    
    def train_and_evaluate(X_train, X_test, Y_train, Y_test, model_name):
        print(f"\n--- 评估: {model_name} ---")
        # 使用 LR 作为最终分类器 (性能一般但稳定)
        lr_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
        lr_model.fit(X_train, Y_train)
        
        Y_pred = lr_model.predict(X_test)
        
        acc = accuracy_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred, average='macro', labels=[-1, 1])
        cm = confusion_matrix(Y_test, Y_pred, labels=[-1, 1])
        
        print(f"Accuracy (整体): {acc:.4f}")
        print(f"Macro F1 (正负情感): {f1:.4f}")
        print(f"混淆矩阵:\n{cm}")
        return f1

    print("\n\n--- 4. 最终特征质量评估（仅AnglE） ---")
    
    # 只评估AnglE-ALCE
    f1_angle = train_and_evaluate(X_train_angle, X_test_angle, Y_train, Y_test, "AnglE-ALCE")

    print("\n\n--- 总结 ---")
    print(f"AnglE-ALCE 模型的 Macro F1 分数: {f1_angle:.4f}")
    
    # 根据F1分数给出评价
    if f1_angle > 0.9:
        print("🎉 性能卓越！您的 AnglE 模型在方面级情感分析上表现出色！")
    elif f1_angle > 0.8:
        print("✅ 性能优秀！AnglE 模型表现良好。")
    elif f1_angle > 0.7:
        print("📊 性能中等，可以考虑进一步优化。")
    else:
        print("⚠️ 性能有待提升，建议检查模型训练或数据质量。")

if __name__ == '__main__':
    evaluate_feature_quality()