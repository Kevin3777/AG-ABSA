# final_evaluation_single_model.py
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
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. 配置区域 (请确认路径是否正确)
# ============================================

# 数据集路径
TRAIN_CSV_FILE = r'asap-master\data\train.csv'
TEST_CSV_FILE = r'asap-master\data\test.csv'

# 模型路径配置
# 指向你刚刚训练完的第5轮Checkpoint
MODEL_PATH_ANGLE_ALCE = r'checkpoints\angle_alce_encoder_final_V2_another_2\checkpoint-655' 

# 将其他模型设为 None 以跳过评估
MODEL_PATH_DOT_PRODUCT = None 
BASE_MODEL_NAME = None 

# 结果保存目录
RESULTS_DIR = r"eval\12_16/pro_v4_evaluation_results"
BATCH_SIZE = 128

# Aspect 映射
ASPECT_MAPPING = {
    'Food#Taste': '口味',
    'Service#Hospitality': '服务态度',
    'Price#Cost_effective': '性价比'
}

# ============================================
# 2. PerfectAnglE 独立实现类
# ============================================
class PerfectAnglE:
    """
    完全匹配AnglE的独立实现
    """
    def __init__(self, model_path, pooling_strategy='cls', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
    
    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def encode(self, texts, to_numpy=True, device=None, 
               normalize_embedding=False, padding='longest', max_length=None,
               embedding_start=0, embedding_size=None, prompt=None):
        
        if device is None:
            device = next(self.model.parameters()).device
        
        encode_max_length = max_length or self.max_length
        
        if prompt is not None:
            texts = [prompt.format(text=text) for text in texts]
        
        if not isinstance(texts, (list, tuple)):
            texts = [texts]
        
        self.model.eval()
        all_embeddings = []
        
        # 内部批处理循环
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=padding,
                    max_length=encode_max_length,
                    truncation=True,
                    return_tensors='pt'
                ).to(device)
                
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                
                # CLS Pooling
                embeddings = last_hidden_state[:, 0, :]
                
                if normalize_embedding:
                    embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
                
                if to_numpy:
                    embeddings = embeddings.cpu().numpy()
                
                all_embeddings.append(embeddings)
        
        if to_numpy:
            return np.concatenate(all_embeddings, axis=0)
        else:
            return torch.cat(all_embeddings, dim=0)
    
    @classmethod
    def from_pretrained(cls, model_path, pooling_strategy='cls', **kwargs):
        return cls(model_path, pooling_strategy=pooling_strategy, max_length=512)

# ============================================
# 3. 工具函数
# ============================================
def setup_logging():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"eval_log_{timestamp}.txt")
    
    logger = logging.getLogger("aspect_evaluation")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def clean_review_text(text):
    import re
    text = str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\\', '', text) 
    text = text.replace('//', ' ')    
    return text.strip()

def load_data_and_filter(file_path, aspect):
    df = pd.read_csv(file_path)
    df['review'] = df['review'].apply(clean_review_text)
    df = df[df[aspect].isin([1, -1])].copy()
    df['label'] = df[aspect].astype(int)
    return df

def create_ap_pair(aspect_chinese_name, review_context):
    return f"Aspect: {aspect_chinese_name}, Context: {review_context}"

def get_embeddings(model_path, df, aspect_name_cn, model_type="angle", batch_size=128):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_type == "angle":
        texts = [create_ap_pair(aspect_name_cn, text) for text in df['review'].tolist()]
        print(f"正在加载 AnglE 模型: {model_path}")
        
        try:
            model = PerfectAnglE.from_pretrained(model_path, pooling_strategy='cls').to(device)
            embeddings = model.encode(
                texts, to_numpy=True, device=device,
                normalize_embedding=False, padding='longest', max_length=512
            )
            return embeddings
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None
            
    else: # bert baseline
        print(f"正在加载 BERT 模型: {model_path}")
        texts = df['review'].tolist()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device)
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT Embedding", leave=False):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings.append(outputs[0][:, 0, :].cpu().numpy())
        return np.concatenate(embeddings, axis=0)

def train_and_evaluate(X_train, X_test, Y_train, Y_test, model_name, logger, aspect):
    if len(Y_train) == 0 or len(Y_test) == 0:
        return None, None
        
    lr_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
    lr_model.fit(X_train, Y_train)
    Y_pred = lr_model.predict(X_test)
    
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average='macro', labels=[-1, 1])
    
    logger.info(f"  [{model_name}] Accuracy: {acc:.4f} | F1-Macro: {f1:.4f}")
    return acc, f1

# ============================================
# 4. 核心评估流程 (支持跳过模型)
# ============================================
def evaluate_single_aspect(aspect_en, aspect_cn, logger):
    logger.info("-" * 60)
    logger.info(f"评估 Aspect: {aspect_en} ({aspect_cn})")
    
    # 1. 加载数据
    df_train = load_data_and_filter(TRAIN_CSV_FILE, aspect_en)
    df_test = load_data_and_filter(TEST_CSV_FILE, aspect_en)
    
    if len(df_train) == 0:
        logger.warning("数据不足，跳过")
        return None

    Y_train = df_train['label'].values
    Y_test = df_test['label'].values
    
    results = {}
    
    # 2. 评估 AnglE (你的新模型)
    if MODEL_PATH_ANGLE_ALCE:
        X_train = get_embeddings(MODEL_PATH_ANGLE_ALCE, df_train, aspect_cn, "angle", BATCH_SIZE)
        X_test = get_embeddings(MODEL_PATH_ANGLE_ALCE, df_test, aspect_cn, "angle", BATCH_SIZE)
        
        if X_train is not None:
            acc, f1 = train_and_evaluate(X_train, X_test, Y_train, Y_test, "AnglE-Pro-V4", logger, aspect_en)
            results['angle_f1'] = f1
    
    # 3. 评估 Dot-Product (可选)
    if MODEL_PATH_DOT_PRODUCT:
        X_train = get_embeddings(MODEL_PATH_DOT_PRODUCT, df_train, aspect_cn, "angle", BATCH_SIZE)
        X_test = get_embeddings(MODEL_PATH_DOT_PRODUCT, df_test, aspect_cn, "angle", BATCH_SIZE)
        if X_train is not None:
            _, f1 = train_and_evaluate(X_train, X_test, Y_train, Y_test, "Dot-Product", logger, aspect_en)
            results['dot_f1'] = f1

    # 4. 评估 BERT Baseline (可选)
    if BASE_MODEL_NAME:
        X_train = get_embeddings(BASE_MODEL_NAME, df_train, aspect_cn, "bert", BATCH_SIZE)
        X_test = get_embeddings(BASE_MODEL_NAME, df_test, aspect_cn, "bert", BATCH_SIZE)
        if X_train is not None:
            _, f1 = train_and_evaluate(X_train, X_test, Y_train, Y_test, "BERT-Base", logger, aspect_en)
            results['bert_f1'] = f1
            
    return results

def main():
    logger = setup_logging()
    logger.info("开始 AnglE-Pro-V4 专项评估")
    logger.info(f"模型路径: {MODEL_PATH_ANGLE_ALCE}")
    
    # 检查模型是否存在
    if not os.path.exists(MODEL_PATH_ANGLE_ALCE):
        logger.error(f"❌ 找不到模型文件: {MODEL_PATH_ANGLE_ALCE}")
        return

    final_scores = []
    
    for aspect_en, aspect_cn in ASPECT_MAPPING.items():
        res = evaluate_single_aspect(aspect_en, aspect_cn, logger)
        if res and 'angle_f1' in res:
            final_scores.append(res['angle_f1'])
    
    if final_scores:
        avg_f1 = sum(final_scores) / len(final_scores)
        logger.info("=" * 60)
        logger.info(f"🚀 最终平均 F1-Macro: {avg_f1:.4f}")
        logger.info("=" * 60)

if __name__ == "__main__":
    main()