#成功

# final_evaluation_single_model.py
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
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到路径，以便导入训练代码中的AnglE类
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================
# 1. 配置区域 (请确认路径是否正确)
# ============================================

# 数据集路径
TRAIN_CSV_FILE = r'asap-master\data\train.csv'
TEST_CSV_FILE = r'asap-master\data\test.csv'

# 模型路径配置 - 指向你刚刚训练完的Checkpoint
MODEL_PATH_ANGLE_ALCE = r'checkpoints/angle_alce_encoder_final_V2_another_no_angle'  # 修改为文件夹路径

# 将其他模型设为 None 以跳过评估
MODEL_PATH_DOT_PRODUCT = None 
BASE_MODEL_NAME = None 

# 结果保存目录
RESULTS_DIR = r"eval/no_angle/evaluation_results"
BATCH_SIZE = 128

# Aspect 映射
# ASPECT_MAPPING = {
#     'Food#Taste': '口味',
#     'Service#Hospitality': '服务态度',
#     'Price#Cost_effective': '性价比'
# }

ASPECT_MAPPING = {
    'Location#Transportation': '交通', 'Location#Downtown': '地段', 'Location#Easy_to_find': '位置易找度',
    'Service#Queue': '排队', 'Service#Hospitality': '服务态度', 'Service#Parking': '停车', 'Service#Timely': '上菜速度',
    'Price#Level': '价格水平', 'Price#Cost_effective': '性价比', 'Price#Discount': '折扣优惠',
    'Ambience#Decoration': '装修装饰', 'Ambience#Noise': '噪音环境', 'Ambience#Space': '空间', 'Ambience#Sanitary': '卫生',
    'Food#Portion': '份量', 'Food#Taste': '口味', 'Food#Appearance': '外观卖相', 'Food#Recommend': '推荐度'
}

# ============================================
# 2. 从训练代码导入或复制AnglE类
# ============================================

def load_angle_model(model_path, pooling_strategy='cls', max_length=512):
    """加载AnglE模型"""
    try:
        # 尝试从训练代码中导入
        try:
            from train_encoder import AnglE  # 假设训练代码文件名为 train_encoder.py
            print(f"✅ 从训练代码导入AnglE类")
        except ImportError:
            print("⚠️ 无法从训练代码导入AnglE类，使用本地实现")
            # 如果在同一个文件中，可以直接定义
            class AnglE(nn.Module):
                """AnglE 模型主类 - 简化版用于推理"""
                
                def __init__(self, model_name_or_path, max_length=512, pooling_strategy="cls"):
                    super().__init__()
                    
                    self.model = AutoModel.from_pretrained(model_name_or_path)
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                    self.max_length = max_length
                    self.pooling_strategy = pooling_strategy
                
                def forward(self, input_ids, attention_mask=None, token_type_ids=None):
                    """前向传播"""
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        output_hidden_states=True
                    )
                    
                    # 获取最后一层隐藏状态
                    last_hidden_state = outputs.last_hidden_state
                    
                    # CLS池化
                    pooled_output = last_hidden_state[:, 0, :]
                    
                    return pooled_output
                
                @staticmethod
                def from_pretrained(model_name_or_path: str,
                                    max_length: int = 512,
                                    pooling_strategy: str = 'cls',
                                    **kwargs):
                    """从预训练模型加载AnglE"""
                    angle = AnglE(model_name_or_path,
                                  max_length=max_length,
                                  pooling_strategy=pooling_strategy,
                                  **kwargs)
                    return angle
                
                def to(self, device):
                    self.model = self.model.to(device)
                    return self
        
        # 加载模型
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
            # 如果已经是numpy数组，转换为numpy
            return np.concatenate(all_embeddings, axis=0)

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
    """加载并过滤数据"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ 加载数据: {file_path}, 形状: {df.shape}")
        
        # 清理文本
        df['review'] = df['review'].apply(clean_review_text)
        
        # 过滤特定aspect的数据
        if aspect in df.columns:
            df = df[df[aspect].isin([1, -1])].copy()
            df['label'] = df[aspect].astype(int)
            print(f"  过滤后: {df.shape} 行, 正样本: {(df['label'] == 1).sum()}, 负样本: {(df['label'] == -1).sum()}")
        else:
            print(f"⚠️ 数据集中没有 {aspect} 列")
            return pd.DataFrame()
            
        return df
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return pd.DataFrame()

def create_ap_pair(aspect_chinese_name, review_context):
    """创建Aspect-Polarity对"""
    return f"Aspect: {aspect_chinese_name}, Context: {review_context}"

def get_embeddings(model_path, df, aspect_name_cn, model_type="angle", batch_size=128):
    """获取文本嵌入"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    if model_type == "angle":
        texts = [create_ap_pair(aspect_name_cn, text) for text in df['review'].tolist()]
        print(f"正在加载 AnglE 模型: {model_path}")
        print(f"文本数量: {len(texts)}")
        
        try:
            # 加载模型
            model = load_angle_model(model_path, pooling_strategy='cls')
            if model is None:
                print(f"❌ 模型加载失败")
                return None
            
            model = model.to(device)
            
            # 编码文本
            print("正在编码文本...")
            embeddings = encode_with_angle(
                model, texts, to_numpy=True, device=device,
                normalize_embedding=False, padding='longest', max_length=512
            )
            
            print(f"✅ 编码完成，嵌入形状: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"❌ 编码失败: {e}")
            import traceback
            print(traceback.format_exc())
            return None
            
    else:  # bert baseline
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
    """训练逻辑回归并评估"""
    if len(Y_train) == 0 or len(Y_test) == 0:
        logger.warning(f"  [{model_name}] 数据不足，跳过")
        return None, None
    
    if X_train is None or X_test is None:
        logger.warning(f"  [{model_name}] 嵌入为空，跳过")
        return None, None
        
    print(f"  [{model_name}] 训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    
    try:
        lr_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
        lr_model.fit(X_train, Y_train)
        Y_pred = lr_model.predict(X_test)
        
        acc = accuracy_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred, average='macro', labels=[-1, 1])
        
        logger.info(f"  [{model_name}] Accuracy: {acc:.4f} | F1-Macro: {f1:.4f}")
        print(f"  [{model_name}] Accuracy: {acc:.4f} | F1-Macro: {f1:.4f}")
        
        # 输出详细分类报告
        precision = precision_score(Y_test, Y_pred, average='macro', labels=[-1, 1])
        recall = recall_score(Y_test, Y_pred, average='macro', labels=[-1, 1])
        cm = confusion_matrix(Y_test, Y_pred, labels=[-1, 1])
        
        logger.info(f"  [{model_name}] Precision: {precision:.4f} | Recall: {recall:.4f}")
        logger.info(f"  [{model_name}] Confusion Matrix:\n{cm}")
        
        return acc, f1
        
    except Exception as e:
        logger.error(f"  [{model_name}] 训练/评估失败: {e}")
        print(f"  [{model_name}] 训练/评估失败: {e}")
        return None, None

# ============================================
# 4. 核心评估流程
# ============================================
def evaluate_single_aspect(aspect_en, aspect_cn, logger, angle_model):
    """评估单个aspect"""
    logger.info("-" * 60)
    logger.info(f"评估 Aspect: {aspect_en} ({aspect_cn})")
    print(f"\n{'='*60}")
    print(f"评估 Aspect: {aspect_en} ({aspect_cn})")
    
    # 1. 加载数据
    print("加载训练数据...")
    df_train = load_data_and_filter(TRAIN_CSV_FILE, aspect_en)
    print("加载测试数据...")
    df_test = load_data_and_filter(TEST_CSV_FILE, aspect_en)
    
    if len(df_train) == 0 or len(df_test) == 0:
        logger.warning("数据不足，跳过")
        return None

    Y_train = df_train['label'].values
    Y_test = df_test['label'].values
    
    results = {}
    
    # 2. 评估 AnglE (你的新模型)
    if angle_model is not None:
        print(f"为 {aspect_cn} 生成AnglE嵌入...")
        # 为训练集生成嵌入
        train_texts = [create_ap_pair(aspect_cn, text) for text in df_train['review'].tolist()]
        X_train = encode_with_angle(
            angle_model, train_texts, to_numpy=True, device='cuda' if torch.cuda.is_available() else 'cpu',
            normalize_embedding=False, padding='longest', max_length=512
        )
        
        # 为测试集生成嵌入
        test_texts = [create_ap_pair(aspect_cn, text) for text in df_test['review'].tolist()]
        X_test = encode_with_angle(
            angle_model, test_texts, to_numpy=True, device='cuda' if torch.cuda.is_available() else 'cpu',
            normalize_embedding=False, padding='longest', max_length=512
        )
        
        if X_train is not None and X_test is not None:
            acc, f1 = train_and_evaluate(X_train, X_test, Y_train, Y_test, "AnglE-Pro-V4", logger, aspect_en)
            results['angle_f1'] = f1
            results['angle_acc'] = acc
    
    return results

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("开始 AnglE-Pro-V4 专项评估")
    logger.info(f"模型路径: {MODEL_PATH_ANGLE_ALCE}")
    print("开始 AnglE-Pro-V4 专项评估")
    print(f"模型路径: {MODEL_PATH_ANGLE_ALCE}")
    
    # 检查模型是否存在
    if not os.path.exists(MODEL_PATH_ANGLE_ALCE):
        logger.error(f"❌ 找不到模型文件: {MODEL_PATH_ANGLE_ALCE}")
        print(f"❌ 找不到模型文件: {MODEL_PATH_ANGLE_ALCE}")
        # 尝试列出目录内容
        parent_dir = os.path.dirname(MODEL_PATH_ANGLE_ALCE)
        if os.path.exists(parent_dir):
            print(f"目录 {parent_dir} 中的文件:")
            for f in os.listdir(parent_dir):
                print(f"  {f}")
        return

    # 加载AnglE模型（一次加载，多次使用）
    angle_model = None
    if MODEL_PATH_ANGLE_ALCE:
        print("加载AnglE模型...")
        angle_model = load_angle_model(MODEL_PATH_ANGLE_ALCE, pooling_strategy='cls')
        if angle_model:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            angle_model = angle_model.to(device)
            print(f"✅ AnglE模型加载成功，使用设备: {device}")
        else:
            print("❌ AnglE模型加载失败")
            return

    final_scores = []
    acc_scores = []
    
    # 评估每个aspect
    for aspect_en, aspect_cn in ASPECT_MAPPING.items():
        print(f"\n评估 {aspect_en} ({aspect_cn})...")
        res = evaluate_single_aspect(aspect_en, aspect_cn, logger, angle_model)
        if res and 'angle_f1' in res:
            final_scores.append(res['angle_f1'])
            if 'angle_acc' in res:
                acc_scores.append(res['angle_acc'])
    
    if final_scores:
        avg_f1 = sum(final_scores) / len(final_scores)
        avg_acc = sum(acc_scores) / len(acc_scores) if acc_scores else 0
        
        logger.info("=" * 60)
        logger.info(f"🚀 最终结果:")
        logger.info(f"  平均 Accuracy: {avg_acc:.4f}")
        logger.info(f"  平均 F1-Macro: {avg_f1:.4f}")
        logger.info("=" * 60)
        
        print("\n" + "="*60)
        print(f"🚀 最终结果:")
        print(f"  平均 Accuracy: {avg_acc:.4f}")
        print(f"  平均 F1-Macro: {avg_f1:.4f}")
        print("="*60)
        
        # 保存结果到文件
        result_file = os.path.join(RESULTS_DIR, "final_results.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"模型路径: {MODEL_PATH_ANGLE_ALCE}\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"平均 Accuracy: {avg_acc:.4f}\n")
            f.write(f"平均 F1-Macro: {avg_f1:.4f}\n")
            f.write("\n各Aspect结果:\n")
            for i, (aspect_en, aspect_cn) in enumerate(ASPECT_MAPPING.items()):
                if i < len(final_scores):
                    f.write(f"  {aspect_cn}: F1={final_scores[i]:.4f}")
                    if i < len(acc_scores):
                        f.write(f", Acc={acc_scores[i]:.4f}")
                    f.write("\n")
        
        print(f"✅ 结果已保存到: {result_file}")

if __name__ == "__main__":
    main()