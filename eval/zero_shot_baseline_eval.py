import pandas as pd
import numpy as np
import requests
import json
import re
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import time

# --- 核心修复：定义 ASPECT_MAPPING 字典 (解决 NameError) ---
ASPECT_MAPPING = {
    'Location#Transportation': '交通', 'Location#Downtown': '地段', 'Location#Easy_to_find': '位置易找度',
    'Service#Queue': '排队', 'Service#Hospitality': '服务态度', 'Service#Parking': '停车', 'Service#Timely': '上菜速度',
    'Price#Level': '价格水平', 'Price#Cost_effective': '性价比', 'Price#Discount': '折扣优惠',
    'Ambience#Decoration': '装修装饰', 'Ambience#Noise': '噪音环境', 'Ambience#Space': '空间', 'Ambience#Sanitary': '卫生',
    'Food#Portion': '份量', 'Food#Taste': '口味', 'Food#Appearance': '外观卖相', 'Food#Recommend': '推荐度'
}

# --- 配置参数 (与 RAG 脚本保持一致) ---
TEST_CSV_FILE = r'asap-master\data\test.csv' # 测试集路径
LLM_MODEL = "THUDM/glm-4-9b-chat"
SILICONFLOW_API_KEY = "sk-mskwuczqlpvmmubgmtcejgolnnsapcumuyyuusgdgjanezfi"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# 评估 Aspect
ASPECT_TO_EVALUATE = 'Food#Taste'
ASPECT_CN = ASPECT_MAPPING[ASPECT_TO_EVALUATE] # 使用字典获取中文名

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

def zero_shot_call_llm(review, aspect_cn):
    """
    Zero-Shot 推理：只提供问题，不提供检索证据。
    Prompt 结构与 RAG-P 保持逻辑一致，确保公平性。
    """
    examples_str = "" # Zero-Shot 模式下，证据为空

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
            match = re.search(r'([+-]?1|0)', content)
            if match:
                val = int(match.group(0))
                return val
        return 0 # API 失败或无法解析时，默认中性
    except Exception:
        return 0

def evaluate_baseline():
    df_test = load_test_data(TEST_CSV_FILE)
    if df_test is None: return

    target_aspect = ASPECT_TO_EVALUATE
    aspect_cn = ASPECT_MAPPING[target_aspect]
    
    # 筛选出有明确标签的样本进行评估 (只评估 1 或 -1)
    test_samples = df_test[df_test[target_aspect].isin([1, -1])].sample(n=1000, random_state=42)
    
    y_true = []
    y_pred = []
    
    print(f"\n--- 开始评估 Zero-Shot LLM 基线 ({aspect_cn}) ---")
    
    for _, row in tqdm(test_samples.iterrows(), total=len(test_samples), desc="Evaluating Baseline Samples"):
        review = row['review']
        true_label = int(row[target_aspect])
        
        # Zero-Shot 推理
        pred_label = zero_shot_call_llm(review, aspect_cn)
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        
        time.sleep(1.5) # 增加 API 延时，避免 429 Rate Limit 错误

    # 计算指标
    labels_to_eval = [1, -1]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=labels_to_eval, average='macro')
    
    print(f"\n\n=== Zero-Shot Baseline 评估结果 ({ASPECT_TO_EVALUATE}) ===")
    print(f"评估样本数: {len(y_true)}")
    print(f"Accuracy (整体): {acc:.4f}")
    print(f"Macro F1 (正负情感): {f1:.4f}")
    print(f"混淆矩阵:\n{confusion_matrix(y_true, y_pred)}")

if __name__ == '__main__':
    evaluate_baseline()