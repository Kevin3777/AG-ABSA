# final_evaluation_perfect_match.py
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
# 完全匹配AnglE的独立实现
# ============================================

class PerfectAnglE:
    """
    完全匹配原始AnglE的独立实现
    使用原始AnglE的默认参数：
    - pooling_strategy='cls'
    - normalize_embedding=False
    - max_length=512（AnglE默认）
    """
    def __init__(self, model_path, pooling_strategy='cls', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.max_length = max_length  # 使用AnglE默认的512
        self.pooling_strategy = pooling_strategy
    
    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def encode(self, texts, to_numpy=True, device=None, 
               normalize_embedding=False, padding='longest', max_length=None,
               embedding_start=0, embedding_size=None, prompt=None):
        """
        完全匹配AnglE.encode方法的参数和实现
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        # 使用传入的max_length或默认值
        encode_max_length = max_length or self.max_length
        
        # 应用提示模板
        if prompt is not None:
            texts = [prompt.format(text=text) for text in texts]
        
        if not isinstance(texts, (list, tuple)):
            texts = [texts]
        
        self.model.eval()
        all_embeddings = []
        
        # 注意：AnglE.encode内部可能没有显式批处理
        # 但我们为了效率使用批处理
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
                
                # 关键：使用CLS池化（AnglE默认）
                embeddings = last_hidden_state[:, 0, :]
                
                # 处理embedding_start和embedding_size
                if embedding_start > 0:
                    embeddings = embeddings[:, embedding_start:]
                
                if embedding_size is not None:
                    embeddings = embeddings[:, :embedding_size]
                
                # 关键：只有当normalize_embedding为True时才归一化
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
        # 注意：AnglE.from_pretrained默认max_length=512
        return cls(model_path, pooling_strategy=pooling_strategy, max_length=512)


# ============================================
# 评估代码（使用完全匹配的实现）
# ============================================

# 配置参数
TRAIN_CSV_FILE = r'asap-master\data\train.csv'
TEST_CSV_FILE = r'asap-master\data\test.csv'
MODEL_PATH_ANGLE_ALCE = r'checkpoints\angle_alce_encoder_final_V2\checkpoint-655' 
MODEL_PATH_DOT_PRODUCT = r'checkpoint_dotproduct\dot_product_model\best_model'
BASE_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"

# Aspect映射
ASPECT_MAPPING = {
    'Food#Taste': '口味',
    'Service#Hospitality': '服务态度',
    'Price#Cost_effective': '性价比'
}

BATCH_SIZE = 128
RESULTS_DIR = "eval\with_bert_independ\evaluation_results"

def setup_logging():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"evaluation_log_{timestamp}.txt")
    
    logger = logging.getLogger("aspect_evaluation")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    """
    关键：使用完全匹配AnglE默认参数的实现
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 准备输入文本
    if model_type == "angle":
        texts = [create_ap_pair(aspect_name_cn, text) for text in df['review'].tolist()]
    else:  # bert
        texts = df['review'].tolist()
    
    if model_type == "angle":
        print(f"加载AnglE模型: {model_path}")
        print(f"使用参数: pooling_strategy='cls', normalize_embedding=False, max_length=512")
        
        try:
            # 关键：使用与原始AnglE完全相同的默认参数
            model = PerfectAnglE.from_pretrained(
                model_path,
                pooling_strategy='cls'  # AnglE默认
            ).to(device)
            
            # 关键：使用AnglE的默认参数
            embeddings = model.encode(
                texts,
                to_numpy=True,
                device=device,
                normalize_embedding=False,  # 关键：AnglE默认False
                padding='longest',
                max_length=512,  # 关键：使用AnglE默认的512，不是192
                prompt=None
            )
            
            print(f"✅ AnglE模型加载成功")
            print(f"   嵌入形状: {embeddings.shape}")
            print(f"   嵌入范数范围: [{np.linalg.norm(embeddings, axis=1).min():.2f}, {np.linalg.norm(embeddings, axis=1).max():.2f}]")
            
        except Exception as e:
            print(f"加载失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 回退方案：直接使用transformers
            print("尝试回退方案...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path).to(device)
            
            embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), 
                         desc=f"提取AnglE嵌入", leave=False):
                batch_texts = texts[i:i + batch_size]
                inputs = tokenizer(
                    batch_texts, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True,
                    max_length=512  # 使用512
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # CLS向量，不归一化
                embeddings.append(outputs[0][:, 0, :].cpu().numpy())
            
            embeddings = np.concatenate(embeddings, axis=0)
            print(f"✅ 使用回退方案提取嵌入")
    
    else:  # bert基线
        print(f"加载BERT模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device)
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), 
                     desc=f"提取BERT嵌入", leave=False):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # CLS向量，不归一化
            embeddings.append(outputs[0][:, 0, :].cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
    
    return embeddings

# 其他函数保持不变...
def train_and_evaluate(X_train, X_test, Y_train, Y_test, model_name, logger, aspect):
    """训练并评估模型"""
    try:
        if len(Y_train) == 0 or len(Y_test) == 0:
            logger.warning(f"{aspect} - {model_name}: 没有足够的样本")
            return None, None, None, None, None, None
        
        lr_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
        lr_model.fit(X_train, Y_train)
        
        Y_pred = lr_model.predict(X_test)
        
        acc = accuracy_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred, average='macro', labels=[-1, 1])
        precision = precision_score(Y_test, Y_pred, average='macro', labels=[-1, 1], zero_division=0)
        recall = recall_score(Y_test, Y_pred, average='macro', labels=[-1, 1], zero_division=0)
        cm = confusion_matrix(Y_test, Y_pred, labels=[-1, 1])
        
        logger.info(f"{aspect} - {model_name} 评估结果:")
        logger.info(f"  准确率: {acc:.4f}")
        logger.info(f"  Macro F1: {f1:.4f}")
        logger.info(f"  精确率: {precision:.4f}")
        logger.info(f"  召回率: {recall:.4f}")
        logger.info(f"  混淆矩阵:\n{cm}")
        logger.info(f"  训练样本数: {len(Y_train)}, 测试样本数: {len(Y_test)}")
        logger.info(f"  正负样本分布 - 训练: {np.bincount(Y_train + 1)}, 测试: {np.bincount(Y_test + 1)}")
        
        return acc, f1, precision, recall, cm, lr_model
        
    except Exception as e:
        logger.error(f"{aspect} - {model_name} 评估失败: {str(e)}")
        return None, None, None, None, None, None

def evaluate_single_aspect(aspect_en, aspect_cn, logger):
    """评估单个aspect"""
    logger.info("="*80)
    logger.info(f"开始评估: {aspect_en} ({aspect_cn})")
    logger.info("="*80)
    
    try:
        # 1. 数据准备
        logger.info("步骤1: 加载和准备数据")
        df_train = load_data_and_filter(TRAIN_CSV_FILE, aspect_en)
        df_test = load_data_and_filter(TEST_CSV_FILE, aspect_en)
        
        train_samples = len(df_train)
        test_samples = len(df_test)
        
        if train_samples == 0 or test_samples == 0:
            logger.warning(f"数据不足: 训练集 {train_samples} 样本, 测试集 {test_samples} 样本")
            return None
        
        logger.info(f"数据统计: 训练集 {train_samples} 样本, 测试集 {test_samples} 样本")
        
        # 2. 特征提取
        logger.info("步骤2: 提取特征向量")
        
        # 2.1 AnglE-ALCE 向量
        logger.info("  提取 AnglE-ALCE 向量...")
        X_train_angle_alce = get_embeddings(
            MODEL_PATH_ANGLE_ALCE, df_train, aspect_cn, 
            model_type="angle", batch_size=BATCH_SIZE
        )
        X_test_angle_alce = get_embeddings(
            MODEL_PATH_ANGLE_ALCE, df_test, aspect_cn, 
            model_type="angle", batch_size=BATCH_SIZE
        )
        
        # 2.2 点积模型向量
        logger.info("  提取点积模型向量...")
        X_train_dot_product = get_embeddings(
            MODEL_PATH_DOT_PRODUCT, df_train, aspect_cn, 
            model_type="angle", batch_size=BATCH_SIZE
        )
        X_test_dot_product = get_embeddings(
            MODEL_PATH_DOT_PRODUCT, df_test, aspect_cn, 
            model_type="angle", batch_size=BATCH_SIZE
        )
        
        # 2.3 BERT-CLS 向量 (基线)
        logger.info("  提取 BERT-CLS 向量...")
        X_train_bert = get_embeddings(
            BASE_MODEL_NAME, df_train, aspect_cn, 
            model_type="bert", batch_size=BATCH_SIZE
        )
        X_test_bert = get_embeddings(
            BASE_MODEL_NAME, df_test, aspect_cn, 
            model_type="bert", batch_size=BATCH_SIZE
        )
        
        Y_train = df_train['label'].values
        Y_test = df_test['label'].values
        
        # 3. 训练和评估
        logger.info("步骤3: 训练和评估模型")
        
        results = {}
        
        # 3.1 评估 AnglE-ALCE
        logger.info("  评估 AnglE-ALCE...")
        results['angle_alce'] = train_and_evaluate(
            X_train_angle_alce, X_test_angle_alce, 
            Y_train, Y_test, "AnglE-ALCE", logger, aspect_en
        )
        
        # 3.2 评估点积模型
        logger.info("  评估点积模型...")
        results['dot_product'] = train_and_evaluate(
            X_train_dot_product, X_test_dot_product,
            Y_train, Y_test, "Dot-Product", logger, aspect_en
        )
        
        # 3.3 评估 BERT-CLS (基线)
        logger.info("  评估 BERT-CLS...")
        results['bert'] = train_and_evaluate(
            X_train_bert, X_test_bert,
            Y_train, Y_test, "BERT-CLS", logger, aspect_en
        )
        
        # 4. 结果对比
        logger.info("步骤4: 结果对比分析")
        
        comparison_data = {
            'aspect_en': aspect_en,
            'aspect_cn': aspect_cn,
            'train_samples': train_samples,
            'test_samples': test_samples
        }
        
        f1_scores = {}
        for model_name, result in results.items():
            if result and result[1] is not None:
                f1_scores[model_name] = result[1]
                
                comparison_data[f'{model_name}_acc'] = result[0]
                comparison_data[f'{model_name}_f1'] = result[1]
                comparison_data[f'{model_name}_precision'] = result[2]
                comparison_data[f'{model_name}_recall'] = result[3]
        
        if len(f1_scores) >= 2:
            best_model = max(f1_scores, key=f1_scores.get)
            best_f1 = f1_scores[best_model]
            
            comparison_data['best_model'] = best_model
            comparison_data['best_f1'] = best_f1
            
            logger.info(f"对比分析:")
            for model, f1 in f1_scores.items():
                logger.info(f"  {model}: F1 = {f1:.4f}")
            
            logger.info(f"  最佳模型: {best_model} (F1 = {best_f1:.4f})")
            
            if 'bert' in f1_scores and 'angle_alce' in f1_scores:
                improvement = f1_scores['angle_alce'] - f1_scores['bert']
                comparison_data['improvement_alce_vs_bert'] = improvement
                logger.info(f"  AnglE-ALCE 相比 BERT 提升: {improvement:.4f}")
            
            if 'bert' in f1_scores and 'dot_product' in f1_scores:
                improvement = f1_scores['dot_product'] - f1_scores['bert']
                comparison_data['improvement_dot_vs_bert'] = improvement
                logger.info(f"  点积模型相比 BERT 提升: {improvement:.4f}")
        
        logger.info(f"完成评估: {aspect_en} ({aspect_cn})")
        return comparison_data
        
    except Exception as e:
        logger.error(f"评估 {aspect_en} 时出现错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def quick_evaluation():
    """快速评估"""
    print("快速评估测试...")
    
    logger = setup_logging()
    logger.info("快速评估测试")
    
    all_results = []
    
    for aspect_en, aspect_cn in ASPECT_MAPPING.items():
        logger.info(f"\n评估: {aspect_en} ({aspect_cn})")
        result = evaluate_single_aspect(aspect_en, aspect_cn, logger)
        if result:
            all_results.append(result)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n快速评估结果:")
        print(results_df[['aspect_en', 'aspect_cn', 'bert_f1', 'dot_product_f1', 'angle_alce_f1']])
        
        # 与原始结果对比
        print("\n与原始结果对比:")
        print("原始结果 - AnglE-ALCE: 91.70%, Dot-Product: 86.51%, BERT-CLS: 84.86%")


if __name__ == '__main__':
    print("完全匹配AnglE的独立评估程序")
    print("="*80)
    print("使用AnglE默认参数:")
    print("  - pooling_strategy='cls'")
    print("  - normalize_embedding=False")
    print("  - max_length=512")
    print("="*80)
    
    # 检查路径和模型
    print("检查模型路径...")
    print(f"AnglE-ALCE 路径: {'✅ 存在' if os.path.exists(MODEL_PATH_ANGLE_ALCE) else '❌ 不存在'}")
    print(f"点积模型路径: {'✅ 存在' if os.path.exists(MODEL_PATH_DOT_PRODUCT) else '❌ 不存在'}")
    
    # 创建结果目录
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    quick_evaluation()