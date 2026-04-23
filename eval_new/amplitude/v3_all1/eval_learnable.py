# final_evaluation_single_model_cv.py
# 修正版：仅使用测试集进行5折交叉验证，消除数据泄露

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

# 添加当前目录到路径，以便导入训练代码中的AnglE类
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================
# 1. 配置区域 (请确认路径是否正确)
# ============================================

# 数据集路径（注意：不再使用训练集）
TEST_CSV_FILE = r'asap-master\data\test.csv'

# 模型路径配置 - 指向你刚刚训练完的Checkpoint
MODEL_PATH_ANGLE_ALCE = r'checkpoints_learnable/v3_all1'  # 修改为文件夹路径

# 结果保存目录
RESULTS_DIR = r"eval_new\learnable\v3_all1\result\evaluation_results_cv"
BATCH_SIZE = 128
N_FOLDS = 5                     # 交叉验证折数
RANDOM_STATE = 42

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

# ASPECT_MAPPING = {
#     'Food#Taste': '口味',
#     'Service#Hospitality': '服务态度',
#     'Price#Cost_effective': '性价比'
# }

# ============================================
# 2. 从训练代码导入或复制AnglE类（保持不变）
# ============================================

def load_angle_model(model_path, pooling_strategy='cls', max_length=512):
    """加载AnglE模型 (包含 AP-Proj 复数投影层)"""
    try:
        # 在评估脚本中独立定义投影层结构
        class ComplexProjection(nn.Module):
            """残差可学习复数空间投影层 (Residual AP-Proj)"""
            def __init__(self, hidden_size):
                super().__init__()
                self.half_size = hidden_size // 2
                
                # 注意这里：输入维度改为了 half_size
                self.proj_re = nn.Linear(self.half_size, self.half_size)
                self.proj_im = nn.Linear(self.half_size, self.half_size)
                
                # 【核心黑科技：零初始化 (Zero-Initialization)】
                # 将权重和偏置全部设为 0。
                # 配合前向传播的残差连接，保证初始状态不对预训练向量造成任何破坏。
                nn.init.zeros_(self.proj_re.weight)
                nn.init.zeros_(self.proj_im.weight)
                nn.init.zeros_(self.proj_re.bias)
                nn.init.zeros_(self.proj_im.bias)
                
            def forward(self, x):
                # 1. 按照基线模型的方式先切分
                re_orig, im_orig = torch.chunk(x, 2, dim=1)
                
                # 2. 残差相加 (Original + Delta)
                # 初始阶段 proj_re(re_orig) 全部是 0，所以 re == re_orig
                re = re_orig + self.proj_re(re_orig)
                im = im_orig + self.proj_im(im_orig)
                # 3. 拼接输出
                return torch.cat([re, im], dim=1)

        # 专用于评估的包装类
        class AnglE_Eval(nn.Module):
            def __init__(self, model_name_or_path, max_length=512):
                super().__init__()
                self.model = AutoModel.from_pretrained(model_name_or_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                self.max_length = max_length
                
                # 动态注入投影层并尝试加载权重
                hidden_size = self.model.config.hidden_size
                self.model.add_module('complex_proj', ComplexProjection(hidden_size))
                
                proj_path = os.path.join(model_name_or_path, 'complex_proj.bin')
                if os.path.exists(proj_path):
                    self.model.complex_proj.load_state_dict(torch.load(proj_path, map_location='cpu'))
                    print(f"✅ 评估脚本成功加载 AP-Proj 投影层权重: {proj_path}")
                else:
                    print("⚠️ 警告: 未找到 complex_proj.bin，这可能是一个旧版模型，将不使用投影层")
                    delattr(self.model, 'complex_proj')
            
            def forward(self, input_ids, attention_mask=None, token_type_ids=None):
                """前向传播"""
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_hidden_states=True
                )
                
                # 获取 CLS 向量
                pooled_output = outputs.last_hidden_state[:, 0, :]
                
                # ====== [关键！如果存在投影层，通过它进行特征映射] ======
                if hasattr(self.model, 'complex_proj'):
                    pooled_output = self.model.complex_proj(pooled_output)
                    
                return pooled_output
            
            def to(self, device):
                self.model = self.model.to(device)
                return self
        
        # 实例化并返回
        angle = AnglE_Eval(model_path, max_length=max_length)
        return angle
        
    except Exception as e:
        print(f"❌ 加载AnglE模型失败: {e}")
        import traceback
        traceback.print_exc()
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
# 3. 工具函数（保持不变，但注意 load_data_and_filter 现在只用于测试集）
# ============================================

def setup_logging():
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
    import re
    text = str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\\', '', text) 
    text = text.replace('//', ' ')    
    return text.strip()

def load_data_and_filter(file_path, aspect):
    """加载并过滤数据（现在只用于测试集）"""
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

# ============================================
# 4. 核心评估函数（修改版：只使用测试集，内部交叉验证）
# ============================================

def evaluate_single_aspect_cv(angle_model, df_test, aspect_cn, logger, n_folds=N_FOLDS):
    """在测试集上进行交叉验证评估"""
    if len(df_test) < n_folds:
        logger.warning(f"  {aspect_cn} 测试样本太少 ({len(df_test)}条)，无法进行{n_folds}折交叉验证，跳过")
        return None, None

    texts = df_test['review'].tolist()
    labels = df_test['label'].values

    # 生成所有文本的嵌入（一次性计算）
    logger.info(f"  生成 {aspect_cn} 的嵌入...")
    ap_texts = [create_ap_pair(aspect_cn, t) for t in texts]
    X = encode_with_angle(
        angle_model, ap_texts, to_numpy=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        normalize_embedding=False, padding='longest', max_length=512
    )
    if X is None:
        logger.error(f"  {aspect_cn} 嵌入生成失败")
        return None, None

    # 交叉验证
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    f1_scores = []
    acc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, labels)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        clf = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        f1 = f1_score(y_val, y_pred, average='macro', labels=[-1, 1])
        acc = accuracy_score(y_val, y_pred)
        f1_scores.append(f1)
        acc_scores.append(acc)
        logger.info(f"    折 {fold+1}: F1={f1:.4f}, Acc={acc:.4f}")

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    mean_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)

    logger.info(f"  {aspect_cn} {n_folds}折CV结果: F1={mean_f1:.4f}±{std_f1:.4f}, Acc={mean_acc:.4f}±{std_acc:.4f}")
    return mean_f1, mean_acc

# ============================================
# 5. 主函数（修改版：只加载测试集）
# ============================================

def main():
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("开始修正后的评估：仅使用测试集进行交叉验证")
    logger.info(f"模型路径: {MODEL_PATH_ANGLE_ALCE}")
    logger.info(f"测试集路径: {TEST_CSV_FILE}")
    logger.info(f"交叉验证折数: {N_FOLDS}")
    logger.info("=" * 60)

    # 检查模型是否存在
    if not os.path.exists(MODEL_PATH_ANGLE_ALCE):
        logger.error(f"❌ 找不到模型文件: {MODEL_PATH_ANGLE_ALCE}")
        parent_dir = os.path.dirname(MODEL_PATH_ANGLE_ALCE)
        if os.path.exists(parent_dir):
            logger.info(f"目录 {parent_dir} 中的文件:")
            for f in os.listdir(parent_dir):
                logger.info(f"  {f}")
        return

    # 加载AnglE模型
    angle_model = None
    print("加载AnglE模型...")
    angle_model = load_angle_model(MODEL_PATH_ANGLE_ALCE, pooling_strategy='cls')
    if angle_model is None:
        logger.error("❌ AnglE模型加载失败")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    angle_model = angle_model.to(device)
    logger.info(f"✅ AnglE模型加载成功，使用设备: {device}")

    aspect_f1 = {}
    aspect_acc = {}

    # 评估每个aspect
    for aspect_en, aspect_cn in ASPECT_MAPPING.items():
        logger.info("-" * 60)
        logger.info(f"评估 Aspect: {aspect_en} ({aspect_cn})")
        print(f"\n{'='*60}")
        print(f"评估 Aspect: {aspect_en} ({aspect_cn})")

        # 只加载测试集数据
        df_test = load_data_and_filter(TEST_CSV_FILE, aspect_en)
        if len(df_test) == 0:
            logger.warning("无有效测试数据，跳过")
            continue

        mean_f1, mean_acc = evaluate_single_aspect_cv(angle_model, df_test, aspect_cn, logger, n_folds=N_FOLDS)
        if mean_f1 is not None:
            aspect_f1[aspect_cn] = mean_f1
            aspect_acc[aspect_cn] = mean_acc

    # 计算总体平均
    if aspect_f1:
        avg_f1 = np.mean(list(aspect_f1.values()))
        avg_acc = np.mean(list(aspect_acc.values()))

        logger.info("=" * 60)
        logger.info("🚀 最终结果（测试集交叉验证）")
        logger.info(f"  平均 F1-Macro: {avg_f1:.4f}")
        logger.info(f"  平均 Accuracy: {avg_acc:.4f}")
        logger.info("=" * 60)

        print("\n" + "="*60)
        print("🚀 最终结果（测试集交叉验证）")
        print(f"  平均 F1-Macro: {avg_f1:.4f}")
        print(f"  平均 Accuracy: {avg_acc:.4f}")
        print("="*60)

        # 保存结果到文件
        result_file = os.path.join(RESULTS_DIR, "final_results_cv.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"模型路径: {MODEL_PATH_ANGLE_ALCE}\n")
            f.write(f"测试集: {TEST_CSV_FILE}\n")
            f.write(f"交叉验证折数: {N_FOLDS}\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"平均 Accuracy: {avg_acc:.4f}\n")
            f.write(f"平均 F1-Macro: {avg_f1:.4f}\n")
            f.write("\n各Aspect详细结果:\n")
            for aspect_cn in aspect_f1.keys():
                f.write(f"  {aspect_cn}: F1={aspect_f1[aspect_cn]:.4f}, Acc={aspect_acc.get(aspect_cn, 0):.4f}\n")

        print(f"✅ 结果已保存到: {result_file}")
    else:
        logger.warning("没有有效的评估结果")

if __name__ == "__main__":
    main()