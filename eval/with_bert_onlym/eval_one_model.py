import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import logging
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import normalize  # 【新增】用于L2归一化
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# --- 1. 定义 AnglE_Res 模型结构 (必须与 train_res_angle.py 完全一致) ---
class AnglE_Res(nn.Module):
    def __init__(self, model_name, pooling_strategy='cls'):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        self.hidden_size = self.backbone.config.hidden_size
        
        # 适配器定义
        self.adapter = nn.Linear(self.hidden_size, self.hidden_size * 2)
        
        # 注意：评估时不需要再次初始化为0，因为我们会加载训练好的权重
        # 但结构定义必须保持一致

    def pooling(self, last_hidden_state, attention_mask):
        if self.pooling_strategy == 'cls':
            return last_hidden_state[:, 0]
        else:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = self.pooling(outputs.last_hidden_state, attention_mask)
        
        # 残差计算逻辑
        delta = self.adapter(pooled)
        delta_real = delta[:, :self.hidden_size]
        delta_imag = delta[:, self.hidden_size:]
        
        real_part = pooled + delta_real
        imag_part = delta_imag
        
        return torch.cat([real_part, imag_part], dim=1)

# --- 配置参数 ---
TRAIN_CSV_FILE = r'asap-master\data\train.csv'
TEST_CSV_FILE = r'asap-master\data\test.csv'

# 【重要】这里修改为你新训练的残差模型路径
MODEL_PATH_TO_EVALUATE = r'checkpoints_pro/v1\checkpoint-epoch-5' 

# Aspect 映射
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

BATCH_SIZE = 64
MODEL_NAME = "AnglE-Res-V1"
RESULTS_DIR = f"eval\{MODEL_NAME}_results"

# --- 日志设置 ---
def setup_logging():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"evaluation_log_{timestamp}.txt")
    
    logger = logging.getLogger("aspect_evaluation")
    logger.setLevel(logging.INFO)
    if logger.handlers: logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh = logging.FileHandler(log_file, encoding='utf-8'); fh.setFormatter(formatter)
    ch = logging.StreamHandler(); ch.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

# --- 数据加载 ---
def clean_review_text(text):
    text = str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return text.strip()

def load_data_and_filter(file_path, aspect):
    df = pd.read_csv(file_path)
    df['review'] = df['review'].apply(clean_review_text)
    df = df[df[aspect].isin([1, -1])].copy()
    df['label'] = df[aspect].astype(int)
    return df

def create_ap_pair(aspect_chinese_name, review_context):
    return f"Aspect: {aspect_chinese_name}, Context: {review_context}"

# --- 核心：提取嵌入 ---
def get_embeddings(model_path, df, aspect_name_cn, batch_size=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    texts = [create_ap_pair(aspect_name_cn, text) for text in df['review'].tolist()]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 初始化 AnglE_Res 模型
    model = AnglE_Res(model_path, pooling_strategy='cls')
    
    # 加载权重
    weights_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到权重: {weights_path}")
        
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"提取嵌入 {aspect_name_cn}", leave=False):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=192, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward
            batch_emb = model(inputs['input_ids'], inputs['attention_mask'])
            embeddings.append(batch_emb.cpu().numpy())
    
    final_embeddings = np.concatenate(embeddings, axis=0)
    
    # 【新增】L2 归一化：对于基于 Cosine/Angle 的方法至关重要
    final_embeddings = normalize(final_embeddings, norm='l2', axis=1)
    
    return final_embeddings

def evaluate_single_aspect(aspect_en, aspect_cn, logger):
    try:
        df_train = load_data_and_filter(TRAIN_CSV_FILE, aspect_en)
        df_test = load_data_and_filter(TEST_CSV_FILE, aspect_en)
        
        if len(df_train) == 0 or len(df_test) == 0: return None
        
        X_train = get_embeddings(MODEL_PATH_TO_EVALUATE, df_train, aspect_cn, BATCH_SIZE)
        X_test = get_embeddings(MODEL_PATH_TO_EVALUATE, df_test, aspect_cn, BATCH_SIZE)
        
        Y_train = df_train['label'].values
        Y_test = df_test['label'].values
        
        # 训练分类器
        clf = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        
        acc = accuracy_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred, average='macro', labels=[-1, 1])
        prec = precision_score(Y_test, Y_pred, average='macro', labels=[-1, 1], zero_division=0)
        rec = recall_score(Y_test, Y_pred, average='macro', labels=[-1, 1], zero_division=0)
        
        logger.info(f"[{aspect_cn}] Acc: {acc:.4f} | F1: {f1:.4f}")
        return {'aspect_en': aspect_en, 'aspect_cn': aspect_cn, 'accuracy': acc, 'macro_f1': f1, 'precision': prec, 'recall': rec}
    except Exception as e:
        logger.error(f"Error {aspect_en}: {e}")
        return None

def main():
    logger = setup_logging()
    logger.info(f"🚀 开始评估残差模型: {MODEL_PATH_TO_EVALUATE}")
    
    if not os.path.exists(MODEL_PATH_TO_EVALUATE):
        print(f"❌ 路径不存在: {MODEL_PATH_TO_EVALUATE}"); return

    results = []
    for en, cn in ASPECT_MAPPING.items():
        res = evaluate_single_aspect(en, cn, logger)
        if res: results.append(res)
        
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(RESULTS_DIR, "final_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 评估完成! 平均 F1: {df['macro_f1'].mean():.4f}")
        print(f"详细结果已保存至: {csv_path}")

if __name__ == '__main__':
    main()