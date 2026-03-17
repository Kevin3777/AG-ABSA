import pandas as pd
import faiss
import pickle
import numpy as np
import torch
import os
import sys
import requests
import json
import time
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import re
from tqdm import tqdm

def setup_path():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设脚本在 D:\WorkSpace\AnglE_yj\rag_build\，需要向上两级找到 AnglE
    project_root = os.path.abspath(os.path.join(current_dir, '..')) 
    angle_root = os.path.join(project_root, 'AnglE')
    
    if os.path.exists(angle_root) and angle_root not in sys.path:
        sys.path.insert(0, angle_root)
        print(f"✅ 已添加 AnglE 根目录到路径: {angle_root}")
    elif angle_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"✅ 已添加项目根目录到路径: {project_root}")

setup_path()


# --- 导入 AnglE ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'AnglE', 'angle_emb'))
from angle_emb import AnglE

# --- 配置参数 ---
TEST_CSV_FILE = r'asap-master\data\test.csv' 
INDEX_DIR = 'rag_index'
MODEL_PATH = r'checkpoints\angle_alce_encoder_final\checkpoint-2605' 
TOP_K = 3 
LLM_MODEL = "THUDM/glm-4-9b-chat"
SILICONFLOW_API_KEY = "sk-mskwuczqlpvmmubgmtcejgolnnsapcumuyyuusgdgjanezfi"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# 评估 Aspect
ASPECT_TO_EVALUATE = 'Food#Taste' 
ASPECT_CN = '口味' 

# ---------------------------------------------------------------------------------------------------

def clean_review_text(text):
    text = str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\\', '', text) 
    text = text.replace('//', ' ')    
    return text.strip()

def load_test_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['review'] = df['review'].apply(clean_review_text)
        return df
    except Exception as e:
        print(f"❌ 错误：测试集加载失败，请检查路径。{e}")
        return None

class RAGPredictor:
    def __init__(self):
        print("正在加载资源...")
        self.device = 'cuda'
        
        # 1. 加载 AnglE
        self.angle = AnglE.from_pretrained(MODEL_PATH, pooling_strategy='cls').to(self.device)
        
        # 2. 加载索引
        self.index = faiss.read_index(os.path.join(INDEX_DIR, 'knowledge_base.index'))
        with open(os.path.join(INDEX_DIR, 'knowledge_base_meta.pkl'), 'rb') as f:
            self.kb_meta = pickle.load(f)
            
        print("资源加载完成。")

    def retrieve(self, query_text, aspect_cn, k=TOP_K):
        angle_input = [f"Aspect: {aspect_cn}, Context: {query_text}"]
        query_vec = self.angle.encode(angle_input, to_numpy=True)
        faiss.normalize_L2(query_vec)
        
        _, indices = self.index.search(query_vec, k)
        
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.kb_meta):
                results.append(self.kb_meta[idx])
        return results

    def call_llm(self, review, aspect_cn, retrieved_examples):
        # 构造 Few-Shot Prompt
        examples_str = ""
        # 强制 LLM 学习标准标签：1 (正面), -1 (负面)
        polarity_map = {1: "正面", -1: "负面"} 
        
        for i, ex in enumerate(retrieved_examples):
            # 只使用核心情感标签，忽略 0/-2
            sentiment = polarity_map.get(ex['polarity'], "不适用")
            if sentiment != "不适用":
                examples_str += f"参考 {i+1}：\"{ex['original_text']}\" -> 标签: {ex['polarity']}\n"
            
        prompt = f"""
你是一个情感分析专家。请根据提供的参考样本，判断目标评论在特定方面的情感极性。

--- 目标任务 ---
评论内容："{review}"
需判断的方面：【{aspect_cn}】

--- 检索到的参考样本 (Few-Shot Evidence) ---
{examples_str}

请**严格**根据评论对方面【{aspect_cn}】的情感倾向，输出一个数字：
- **如果情感倾向为正面，输出 1**
- **如果情感倾向为负面，输出 -1**
- **如果评论没有提及该方面，或情感模糊，输出 0**

输出格式：[标签: X] (X为 1, -1, 或 0)
"""
        headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1, 
            "max_tokens": 10 
        }
        
        try:
            response = requests.post(SILICONFLOW_API_URL, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content'].strip()
                # 尝试提取 [标签: X] 中的数字
                match = re.search(r'[+-]?\d+', content)
                if match:
                    val = int(match.group(0))
                    # 评估只关注 1, -1, 0
                    if val in [1, -1, 0]:
                        return val
            return 0
        except Exception:
            return 0 # API 失败，默认中性

def evaluate():
    df_test = load_test_data(TEST_CSV_FILE)
    if df_test is None: return

    predictor = RAGPredictor()
    
    # 筛选出有明确标签的样本进行评估 (排除 -2)
    test_samples = df_test[df_test[ASPECT_TO_EVALUATE] != -2]
    # 🌟 统一评估样本集：只评估有明确情感（1 或 -1）的样本
    test_samples = test_samples[test_samples[ASPECT_TO_EVALUATE].isin([1, -1])].sample(n=1000, random_state=42)
    
    y_true = []
    y_pred = []
    
    print(f"\n评估 Aspect: {ASPECT_CN} ({ASPECT_TO_EVALUATE}) - 共 {len(test_samples)} 条明确情感样本...")
    
    for _, row in tqdm(test_samples.iterrows(), total=len(test_samples), desc="Evaluating Samples"):
        review = row['review']
        true_label = int(row[ASPECT_TO_EVALUATE])
        
        examples = predictor.retrieve(review, ASPECT_CN)
        pred_label = predictor.call_llm(review, ASPECT_CN, examples)
        
        # 评估只记录 1 和 -1 的样本
        y_true.append(true_label)
        # 如果 LLM 输出了 0，我们将其视为错误分类，但保留原始输出
        y_pred.append(pred_label) 
        
        time.sleep(1.5) # 增加 API 延时

    # 计算指标 (只对 1 和 -1 的样本进行评估)
    # F1 score 计算只针对 1 和 -1
    labels_to_eval = [1, -1]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=labels_to_eval, average='macro')
    
    print(f"\n\n=== RAG-P 评估结果 ({ASPECT_TO_EVALUATE}) ===")
    print(f"评估样本数: {len(y_true)}")
    print(f"Accuracy (整体): {acc:.4f}")
    print(f"Macro F1 (正负情感): {f1:.4f}")
    print(f"混淆矩阵:\n{confusion_matrix(y_true, y_pred)}")


if __name__ == '__main__':
    # ⚠️ 请确认 LLM API Key 正常工作，并确保已安装 faiss-cpu
    evaluate()