# final_evaluation_cv.py
# 修正后的评估代码：仅使用测试集进行5折交叉验证，避免数据泄露

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import json
import logging
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到路径，以便导入训练代码中的AnglE类（如果需要）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================
# 1. 配置区域 (请根据实际情况修改)
# ============================================

# 测试集路径（模型训练时未使用过的数据）
TEST_CSV_FILE = r'asap-master\data\test.csv'

# 模型路径配置 - 指向你训练好的Checkpoint
MODEL_PATH_ANGLE = r'checkpoints_simcse/v1'  # 修改为你的模型路径

# 结果保存目录
RESULTS_DIR = r"eval_new\simcse\result\evaluation_results_cv"
BATCH_SIZE = 128
N_FOLDS = 5                # 交叉验证折数
RANDOM_STATE = 42

# Aspect 映射（根据你的任务修改）
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

# ============================================
# 2. 从训练代码导入或定义AnglE类（与之前相同）
# ============================================

def load_angle_model(model_path, pooling_strategy='cls', max_length=512):
    """加载AnglE模型（支持本地实现或从训练代码导入）"""
    try:
        # 尝试从训练代码导入
        try:
            from train_encoder import AnglE  # 假设训练代码文件名为 train_encoder.py
            print(f"✅ 从训练代码导入AnglE类")
        except ImportError:
            print("⚠️ 无法从训练代码导入AnglE类，使用本地实现")
            # 本地实现（简化版）
            class AnglE(nn.Module):
                def __init__(self, model_name_or_path, max_length=512, pooling_strategy="cls"):
                    super().__init__()
                    self.model = AutoModel.from_pretrained(model_name_or_path)
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                    self.max_length = max_length
                    self.pooling_strategy = pooling_strategy
                
                def forward(self, input_ids, attention_mask=None, token_type_ids=None):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        output_hidden_states=True
                    )
                    last_hidden_state = outputs.last_hidden_state
                    pooled_output = last_hidden_state[:, 0, :]  # CLS池化
                    return pooled_output
                
                @staticmethod
                def from_pretrained(model_name_or_path: str,
                                    max_length: int = 512,
                                    pooling_strategy: str = 'cls',
                                    **kwargs):
                    angle = AnglE(model_name_or_path,
                                  max_length=max_length,
                                  pooling_strategy=pooling_strategy,
                                  **kwargs)
                    return angle
                
                def to(self, device):
                    self.model = self.model.to(device)
                    return self
        
        angle = AnglE.from_pretrained(
            model_name_or_path=model_path,
            max_length=max_length,
            pooling_strategy=pooling_strategy
        )
        return angle
    except Exception as e:
        print(f"❌ 加载AnglE模型失败: {e}")
        return None

def encode_with_angle(model, texts, to_numpy=True, device=None, 
                     normalize_embedding=False, padding='longest', max_length=None,
                     prompt=None):
    """使用AnglE模型编码文本"""
    if device is None:
        device = next(model.model.parameters()).device
    
    encode_max_length = max_length or model.max_length
    
    if prompt is not None:
        texts = [prompt.format(text=text) for text in texts]
    
    if not isinstance(texts, (list, tuple)):
        texts = [texts]
    
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            
            inputs = model.tokenizer(
                batch_texts,
                padding=padding,
                max_length=encode_max_length,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            embeddings = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            if normalize_embedding:
                embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
            
            if to_numpy:
                embeddings = embeddings.cpu().numpy()
            
            all_embeddings.append(embeddings)
    
    if to_numpy:
        return np.concatenate(all_embeddings, axis=0)
    else:
        if isinstance(all_embeddings[0], torch.Tensor):
            return torch.cat(all_embeddings, dim=0)
        else:
            return np.concatenate(all_embeddings, axis=0)

# ============================================
# 3. 工具函数
# ============================================

def setup_logging():
    """设置日志"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"eval_log_{timestamp}.txt")
    
    logger = logging.getLogger("aspect_evaluation_cv")
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
    """清理评论文本"""
    import re
    text = str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\\', '', text) 
    text = text.replace('//', ' ')    
    return text.strip()

def load_test_data(file_path, aspect):
    """加载测试集并过滤特定aspect的数据"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ 加载测试集: {file_path}, 形状: {df.shape}")
        
        # 清理文本
        df['review'] = df['review'].apply(clean_review_text)
        
        # 过滤特定aspect的数据（只保留标签为1或-1的样本）
        if aspect in df.columns:
            df = df[df[aspect].isin([1, -1])].copy()
            df['label'] = df[aspect].astype(int)
            print(f"  过滤后: {df.shape} 行, 正样本: {(df['label'] == 1).sum()}, 负样本: {(df['label'] == -1).sum()}")
        else:
            print(f"⚠️ 测试集中没有 {aspect} 列")
            return pd.DataFrame()
        
        return df
    except Exception as e:
        print(f"❌ 加载测试集失败: {e}")
        return pd.DataFrame()

def create_ap_pair(aspect_chinese_name, review_context):
    """创建Aspect-Polarity对（与训练时一致）"""
    return f"Aspect: {aspect_chinese_name}, Context: {review_context}"

# ============================================
# 4. 核心评估函数：在测试集上进行交叉验证
# ============================================

def evaluate_aspect_with_cv(model, df_test, aspect_cn, logger):
    """
    在测试集上进行交叉验证，评估嵌入质量
    返回该aspect的平均F1和平均准确率
    """
    if len(df_test) < 2:
        logger.warning(f"  {aspect_cn} 测试样本太少 ({len(df_test)}条)，无法进行交叉验证，跳过")
        return None, None
    
    texts = df_test['review'].tolist()
    labels = df_test['label'].values
    
    # 生成嵌入
    logger.info(f"  生成 {aspect_cn} 的嵌入...")
    try:
        # 构造Aspect-Polarity对（与训练时使用的prompt一致）
        ap_texts = [create_ap_pair(aspect_cn, t) for t in texts]
        X = encode_with_angle(
            model, ap_texts, to_numpy=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            normalize_embedding=False,
            padding='longest',
            max_length=512
        )
        if X is None:
            logger.error(f"  {aspect_cn} 嵌入生成失败")
            return None, None
    except Exception as e:
        logger.error(f"  {aspect_cn} 嵌入生成异常: {e}")
        return None, None
    
    # 交叉验证
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    f1_scores = []
    acc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, labels)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # 训练逻辑回归
        clf = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        
        # 计算指标
        f1 = f1_score(y_val, y_pred, average='macro', labels=[-1, 1])
        acc = accuracy_score(y_val, y_pred)
        f1_scores.append(f1)
        acc_scores.append(acc)
        
        logger.info(f"    折 {fold+1}: F1={f1:.4f}, Acc={acc:.4f}")
    
    # 汇总
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    mean_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)
    
    logger.info(f"  {aspect_cn} {N_FOLDS}折CV结果: F1={mean_f1:.4f}±{std_f1:.4f}, Acc={mean_acc:.4f}±{std_acc:.4f}")
    
    return mean_f1, mean_acc

# ============================================
# 5. 主函数
# ============================================

def main():
    logger = setup_logging()
    logger.info("="*60)
    logger.info("开始修正后的评估：仅使用测试集进行交叉验证")
    logger.info(f"模型路径: {MODEL_PATH_ANGLE}")
    logger.info(f"测试集路径: {TEST_CSV_FILE}")
    logger.info(f"交叉验证折数: {N_FOLDS}")
    logger.info("="*60)
    
    # 检查模型路径
    if not os.path.exists(MODEL_PATH_ANGLE):
        logger.error(f"❌ 找不到模型文件: {MODEL_PATH_ANGLE}")
        # 尝试列出父目录内容辅助调试
        parent_dir = os.path.dirname(MODEL_PATH_ANGLE)
        if os.path.exists(parent_dir):
            logger.info(f"目录 {parent_dir} 中的文件:")
            for f in os.listdir(parent_dir):
                logger.info(f"  {f}")
        return
    
    # 加载模型
    logger.info("加载AnglE模型...")
    angle_model = load_angle_model(MODEL_PATH_ANGLE, pooling_strategy='cls')
    if angle_model is None:
        logger.error("❌ 模型加载失败，终止评估")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    angle_model = angle_model.to(device)
    logger.info(f"✅ 模型加载成功，使用设备: {device}")
    
    # 存储各aspect结果
    aspect_f1 = {}
    aspect_acc = {}
    
    # 遍历所有aspect
    for aspect_en, aspect_cn in ASPECT_MAPPING.items():
        logger.info("-"*60)
        logger.info(f"评估 Aspect: {aspect_en} ({aspect_cn})")
        
        # 加载测试集数据（只过滤该aspect）
        df_test = load_test_data(TEST_CSV_FILE, aspect_en)
        if len(df_test) == 0:
            logger.warning(f"  无有效测试数据，跳过")
            continue
        
        # 进行交叉验证评估
        mean_f1, mean_acc = evaluate_aspect_with_cv(angle_model, df_test, aspect_cn, logger)
        
        if mean_f1 is not None:
            aspect_f1[aspect_cn] = mean_f1
            aspect_acc[aspect_cn] = mean_acc
    
    # 计算总体平均
    if aspect_f1:
        avg_f1 = np.mean(list(aspect_f1.values()))
        avg_acc = np.mean(list(aspect_acc.values()))
        
        logger.info("="*60)
        logger.info("🚀 最终评估结果（测试集交叉验证）")
        logger.info(f"  平均 F1-Macro: {avg_f1:.4f}")
        logger.info(f"  平均 Accuracy: {avg_acc:.4f}")
        logger.info("="*60)
        
        # 保存结果到文件
        result_file = os.path.join(RESULTS_DIR, "final_results_cv.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"模型路径: {MODEL_PATH_ANGLE}\n")
            f.write(f"测试集: {TEST_CSV_FILE}\n")
            f.write(f"交叉验证折数: {N_FOLDS}\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"平均 Accuracy: {avg_acc:.4f}\n")
            f.write(f"平均 F1-Macro: {avg_f1:.4f}\n")
            f.write("\n各Aspect详细结果:\n")
            for aspect_cn in aspect_f1.keys():
                f.write(f"  {aspect_cn}: F1={aspect_f1[aspect_cn]:.4f}, Acc={aspect_acc.get(aspect_cn, 0):.4f}\n")
        
        logger.info(f"✅ 结果已保存到: {result_file}")
        
        # 同时在控制台打印详细表格
        print("\n各Aspect结果:")
        for aspect_cn in aspect_f1.keys():
            print(f"  {aspect_cn}: F1={aspect_f1[aspect_cn]:.4f}, Acc={aspect_acc.get(aspect_cn, 0):.4f}")
    else:
        logger.warning("没有有效的评估结果")

if __name__ == "__main__":
    main()