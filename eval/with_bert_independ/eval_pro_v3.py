import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import logging
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# --- 1. 配置参数 ---
# 数据路径 (请确认路径是否正确)
TRAIN_CSV_FILE = r'asap-master\data\train.csv'
TEST_CSV_FILE = r'asap-master\data\test.csv'

# 刚刚训练好的模型路径
MODEL_DIR = r'checkpoints_pro_v3\checkpoint-epoch-4' 
ADAPTER_PATH = os.path.join(MODEL_DIR, 'complex_adapter.pt')

# 评估设置
BATCH_SIZE = 64
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

# --- 2. 重新定义模型结构 (必须与训练时一致) ---
class AnglE_Pro(nn.Module):
    def __init__(self, model_path, pooling='cls'):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path)
        self.pooling = pooling
        hidden_size = self.backbone.config.hidden_size
        
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.Tanh()
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(input_ids, attention_mask, token_type_ids=token_type_ids)
        last_hidden = outputs.last_hidden_state
        if self.pooling == 'cls':
            embedding = last_hidden[:, 0]
        else:
            # ... (同上) ...
            pass
            
        adjustment = self.adapter(embedding)
        projected = embedding + self.gate * adjustment
        return F.normalize(projected, p=2, dim=1)

# 加载权重部分
checkpoint = torch.load(os.path.join(MODEL_DIR, "adapter_residuals.pt"))
model.adapter.load_state_dict(checkpoint['adapter'])
model.gate.data = checkpoint['gate']

# --- 3. 工具函数 ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("Eval")

def clean_text(text):
    return str(text).strip()

def load_data(file_path, aspect_en):
    """加载并筛选特定 Aspect 的数据"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return pd.DataFrame()
        
    df = pd.read_csv(file_path)
    df['review'] = df['review'].apply(clean_text)
    # 只保留标签为 1 或 -1 的数据
    df = df[df[aspect_en].isin([1, -1])].copy()
    return df

def get_embeddings(model, tokenizer, texts, aspect_cn, batch_size=64):
    """
    生成 Embedding
    注意：这里模拟训练时的输入格式 -> tokenizer(aspect, context)
    """
    model.eval()
    all_embeddings = []
    
    # 构造 Pair 输入: (Aspect名称, 评论文本)
    # 这样能利用 Token Type IDs 区分 Aspect 和 Context
    
    total = len(texts)
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # 关键：Aspect 作为第一个句子，Review 作为第二个句子
            # 输入变成: [CLS] 交通 [SEP] 这里交通很方便... [SEP]
            batch_aspects = [aspect_cn] * len(batch_texts)
            
            inputs = tokenizer(
                batch_aspects,    # Sentence A
                batch_texts,      # Sentence B
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors='pt'
            ).to(device)
            
            # 模型推理
            embeddings = model(**inputs)
            all_embeddings.append(embeddings.cpu().numpy())
            
    return np.concatenate(all_embeddings, axis=0)

# --- 4. 主评估逻辑 ---
def main():
    logger = setup_logging()
    logger.info(f"🚀 开始评估模型: {MODEL_DIR}")
    
    # 1. 加载模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AnglE_Pro(MODEL_DIR).to(device)
        
        # 加载投影层权重
        if os.path.exists(ADAPTER_PATH):
            logger.info(f"Loading adapter weights from {ADAPTER_PATH}")
            model.complex_adapter.load_state_dict(torch.load(ADAPTER_PATH))
        else:
            logger.warning("⚠️ 未找到 complex_adapter.pt，将使用随机初始化的投影层！(这会导致效果很差)")
            
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return

    # 2. 遍历所有 Aspect 进行评估
    results_list = []
    
    print("\n" + "="*60)
    print(f"{'Aspect (EN)':<30} | {'Aspect (CN)':<10} | {'Acc':<8} | {'F1':<8}")
    print("="*60)
    
    total_f1 = 0
    count = 0
    
    for aspect_en, aspect_cn in ASPECT_MAPPING.items():
        # 加载数据
        train_df = load_data(TRAIN_CSV_FILE, aspect_en)
        test_df = load_data(TEST_CSV_FILE, aspect_en)
        
        if len(train_df) < 10 or len(test_df) < 5:
            continue
            
        # 提取特征
        X_train = get_embeddings(model, tokenizer, train_df['review'].tolist(), aspect_cn)
        X_test = get_embeddings(model, tokenizer, test_df['review'].tolist(), aspect_cn)
        
        Y_train = train_df[aspect_en].values
        Y_test = test_df[aspect_en].values
        
        # 训练分类器 (Logistic Regression)
        clf = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        
        # 计算指标
        acc = accuracy_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred, average='macro') # Macro F1 关注正负类的平衡
        
        print(f"{aspect_en:<30} | {aspect_cn:<10} | {acc:.4f}   | {f1:.4f}")
        
        results_list.append({
            'Aspect': aspect_en,
            'CN': aspect_cn,
            'Accuracy': acc,
            'F1': f1,
            'Train_Size': len(train_df),
            'Test_Size': len(test_df)
        })
        
        total_f1 += f1
        count += 1
        
    # 3. 汇总
    print("="*60)
    if count > 0:
        avg_f1 = total_f1 / count
        print(f"🔥 平均 Macro-F1: {avg_f1:.4f}")
        print("="*60)
        
        # 保存结果
        res_df = pd.DataFrame(results_list)
        save_file = r"eval\with_bert\evaluation_results"
        res_df.to_csv(save_file, index=False, encoding='utf-8-sig')
        logger.info(f"结果已保存至 {save_file}")
    else:
        logger.warning("没有足够的数据进行评估。")

if __name__ == "__main__":
    main()