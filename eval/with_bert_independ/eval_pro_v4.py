import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import logging
import gc
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

# ==========================================
# 1. 配置参数
# ==========================================
# 原始数据路径
TRAIN_CSV_FILE = r'asap-master\data\train.csv'
TEST_CSV_FILE = r'asap-master\data\test.csv'

# 【关键】这里指向你训练好的 v4 模型路径
MODEL_DIR = r'checkpoints_pro_v4\checkpoint-v4-epoch-4' 

# 评估设置
BATCH_SIZE = 32
MAX_LENGTH = 192
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Aspect 映射表
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

# ==========================================
# 2. 模型定义 (v4 Standard 版本)
# ==========================================
class AnglE_Standard_Eval(nn.Module):
    def __init__(self, model_path, pooling='cls'):
        super().__init__()
        # 直接加载 BERT，没有额外的 Adapter
        self.backbone = AutoModel.from_pretrained(model_path)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        last_hidden = outputs.last_hidden_state
        
        # Pooling
        if self.pooling == 'cls':
            embedding = last_hidden[:, 0]
        else:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            embedding = torch.sum(last_hidden * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
        # 标准 AnglE v4 只有归一化，没有投影
        return F.normalize(embedding, p=2, dim=1)

# ==========================================
# 3. 工具函数
# ==========================================
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler()])
    return logging.getLogger("Eval_v4")

def clean_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def load_data(file_path, aspect_en):
    if not os.path.exists(file_path): return pd.DataFrame()
    df = pd.read_csv(file_path)
    df['review'] = df['review'].apply(lambda x: str(x).strip())
    df = df[df[aspect_en].isin([1, -1])].copy()
    return df

def get_embeddings(model, tokenizer, texts, aspect_cn):
    model.eval()
    all_embeddings = []
    total = len(texts)
    
    with torch.no_grad():
        for i in range(0, total, BATCH_SIZE):
            batch_texts = texts[i : i + BATCH_SIZE]
            # 构造输入：Sentence A = Aspect, Sentence B = Context
            batch_aspects = [aspect_cn] * len(batch_texts)
            
            inputs = tokenizer(
                batch_aspects,    
                batch_texts,      
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors='pt'
            ).to(device)
            
            emb = model(**inputs)
            all_embeddings.append(emb.cpu().numpy())
            
    return np.concatenate(all_embeddings, axis=0)

# ==========================================
# 4. 主流程
# ==========================================
def main():
    logger = setup_logging()
    logger.info(f"🚀 开始评估 AnglE Standard (v4)")
    logger.info(f"📂 模型路径: {MODEL_DIR}")
    
    clean_gpu()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AnglE_Standard_Eval(MODEL_DIR).to(device)
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return

    print("\n" + "="*75)
    print(f"{'Aspect (EN)':<30} | {'Aspect (CN)':<10} | {'Acc':<8} | {'F1':<8} | {'Samples'}")
    print("="*75)
    
    total_f1 = 0
    valid_count = 0
    results_list = []
    
    for aspect_en, aspect_cn in ASPECT_MAPPING.items():
        train_df = load_data(TRAIN_CSV_FILE, aspect_en)
        test_df = load_data(TEST_CSV_FILE, aspect_en)
        
        if len(train_df) < 5 or len(test_df) < 5: continue
            
        X_train = get_embeddings(model, tokenizer, train_df['review'].tolist(), aspect_cn)
        X_test = get_embeddings(model, tokenizer, test_df['review'].tolist(), aspect_cn)
        
        Y_train = train_df[aspect_en].values
        Y_test = test_df[aspect_en].values
        
        clf = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        
        acc = accuracy_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred, average='macro') 
        
        print(f"{aspect_en:<30} | {aspect_cn:<10} | {acc:.4f}   | {f1:.4f}   | {len(Y_test)}")
        
        results_list.append({'Aspect': aspect_en, 'F1': f1, 'Acc': acc})
        total_f1 += f1
        valid_count += 1
        
    print("="*75)
    if valid_count > 0:
        print(f"🔥 平均 Macro-F1: {total_f1 / valid_count:.4f}")
        pd.DataFrame(results_list).to_csv(f"eval_res_v4_{datetime.now().strftime('%H%M')}.csv", index=False)

if __name__ == "__main__":
    main()