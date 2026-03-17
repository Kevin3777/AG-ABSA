import pandas as pd
import numpy as np
import torch
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

# --- 🛠️ 路径自动加固 (必须放在所有 import 之前) ---
def setup_path():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设脚本在 D:\WorkSpace\AnglE_yj\rag_build\，需要向上两级找到 AnglE
    project_root = r'D:\WorkSpace\AnglE_yj' 
    angle_root = os.path.join(project_root, 'AnglE')
    
    if os.path.exists(angle_root) and angle_root not in sys.path:
        sys.path.insert(0, angle_root)
        print(f"✅ 已添加 AnglE 根目录到路径: {angle_root}")
    elif angle_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"✅ 已添加项目根目录到路径: {project_root}")

setup_path()

from angle_emb import AnglE


# --- 配置参数 (与您的环境保持一致) ---
TRAIN_CSV_FILE = r'asap-master\data\train.csv'
TEST_CSV_FILE = r'asap-master\data\test.csv'

# 模型路径
MODEL_PATH_ANGLE_ALCE = r'checkpoints\angle_alce_encoder_final_V2\checkpoint-655' 
MODEL_PATH_DOT_PRODUCT = r'checkpoint_dotproduct\dot_product_model\best_model'
BASE_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"  # 基线模型

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

# 评估参数
BATCH_SIZE = 128
RESULTS_DIR = "eval\with_bert\evaluation_results"

# --- 日志设置 ---
def setup_logging():
    """设置日志记录"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"evaluation_log_{timestamp}.txt")
    
    # 创建日志记录器
    logger = logging.getLogger("aspect_evaluation")
    logger.setLevel(logging.INFO)
    
    # 清除之前的处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# --- 数据加载函数 ---
def clean_review_text(text):
    """清理评论文本"""
    import re
    text = str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\\', '', text) 
    text = text.replace('//', ' ')    
    return text.strip()

def load_data_and_filter(file_path, aspect):
    """加载数据并过滤指定aspect的样本"""
    df = pd.read_csv(file_path)
    df['review'] = df['review'].apply(clean_review_text)
    
    # 只保留有明确情感（1 或 -1）的样本
    df = df[df[aspect].isin([1, -1])].copy()
    df['label'] = df[aspect].astype(int)
    
    return df

def create_ap_pair(aspect_chinese_name, review_context):
    """创建 AnglE 所需的 AP Pair 输入格式"""
    return f"Aspect: {aspect_chinese_name}, Context: {review_context}"

def get_embeddings(model_path, df, aspect_name_cn, model_type="angle", batch_size=128):
    """
    提取模型嵌入向量
    
    Args:
        model_path: 模型路径
        df: 数据DataFrame
        aspect_name_cn: 中文aspect名称
        model_type: 模型类型 ("angle", "bert")
        batch_size: 批处理大小
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 准备输入文本
    if model_type == "angle":
        texts = [create_ap_pair(aspect_name_cn, text) for text in df['review'].tolist()]
    else:  # bert
        texts = df['review'].tolist()
    
    # 加载模型和提取嵌入
    if model_type == "angle":
        # 加载AnglE模型
        model = AnglE.from_pretrained(model_path, pooling_strategy='cls').to(device)
        
        # 提取嵌入
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), 
                     desc=f"提取AnglE嵌入", leave=False):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = model.encode(batch_texts, to_numpy=True)
            embeddings.append(batch_embeddings)
    
    else:  # bert
        # 加载BERT模型
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device)
        
        # 提取嵌入
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
            
            # CLS 向量提取
            embeddings.append(outputs[0][:, 0, :].cpu().numpy())
    
    return np.concatenate(embeddings, axis=0)

def train_and_evaluate(X_train, X_test, Y_train, Y_test, model_name, logger, aspect):
    """训练并评估模型"""
    try:
        # 检查样本数量
        if len(Y_train) == 0 or len(Y_test) == 0:
            logger.warning(f"{aspect} - {model_name}: 没有足够的样本")
            return None, None, None, None, None, None
        
        # 训练LR分类器
        lr_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
        lr_model.fit(X_train, Y_train)
        
        # 预测
        Y_pred = lr_model.predict(X_test)
        
        # 计算指标
        acc = accuracy_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred, average='macro', labels=[-1, 1])
        precision = precision_score(Y_test, Y_pred, average='macro', labels=[-1, 1], zero_division=0)
        recall = recall_score(Y_test, Y_pred, average='macro', labels=[-1, 1], zero_division=0)
        cm = confusion_matrix(Y_test, Y_pred, labels=[-1, 1])
        
        # 详细日志
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
        
        # 提取F1分数进行对比
        f1_scores = {}
        for model_name, result in results.items():
            if result[1] is not None:  # F1分数
                f1_scores[model_name] = result[1]
                
                # 保存所有指标
                comparison_data[f'{model_name}_acc'] = result[0]
                comparison_data[f'{model_name}_f1'] = result[1]
                comparison_data[f'{model_name}_precision'] = result[2]
                comparison_data[f'{model_name}_recall'] = result[3]
        
        if len(f1_scores) >= 2:
            # 找出最佳模型
            best_model = max(f1_scores, key=f1_scores.get)
            best_f1 = f1_scores[best_model]
            
            comparison_data['best_model'] = best_model
            comparison_data['best_f1'] = best_f1
            
            logger.info(f"对比分析:")
            for model, f1 in f1_scores.items():
                logger.info(f"  {model}: F1 = {f1:.4f}")
            
            logger.info(f"  最佳模型: {best_model} (F1 = {best_f1:.4f})")
            
            # 计算提升幅度
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

def evaluate_all_aspects():
    """评估所有18个aspect"""
    print("="*80)
    print("开始评估所有 18 个方面")
    print("="*80)
    
    # 设置日志
    logger = setup_logging()
    
    logger.info("模型评估配置:")
    logger.info(f"AnglE-ALCE 模型: {MODEL_PATH_ANGLE_ALCE}")
    logger.info(f"点积模型: {MODEL_PATH_DOT_PRODUCT}")
    logger.info(f"基线模型: {BASE_MODEL_NAME}")
    logger.info(f"批处理大小: {BATCH_SIZE}")
    logger.info(f"评估方面数: {len(ASPECT_MAPPING)}")
    logger.info("="*80)
    
    # 存储所有结果
    all_results = []
    successful_aspects = 0
    
    # 遍历所有aspect进行评估
    for i, (aspect_en, aspect_cn) in enumerate(tqdm(ASPECT_MAPPING.items(), 
                                                    desc="评估所有方面", 
                                                    total=len(ASPECT_MAPPING))):
        logger.info(f"\n评估进度: {i+1}/{len(ASPECT_MAPPING)}")
        
        result = evaluate_single_aspect(aspect_en, aspect_cn, logger)
        
        if result:
            all_results.append(result)
            successful_aspects += 1
    
    # 汇总结果
    logger.info("\n" + "="*80)
    logger.info("评估完成！")
    logger.info(f"成功评估: {successful_aspects}/{len(ASPECT_MAPPING)} 个方面")
    logger.info("="*80)
    
    # 保存详细结果到CSV
    if all_results:
        # 创建DataFrame
        results_df = pd.DataFrame(all_results)
        
        # 重新排列列顺序
        column_order = ['aspect_en', 'aspect_cn', 'train_samples', 'test_samples', 
                       'best_model', 'best_f1']
        
        # 添加模型指标列
        for model in ['bert', 'dot_product', 'angle_alce']:
            for metric in ['acc', 'f1', 'precision', 'recall']:
                column_name = f'{model}_{metric}'
                if column_name in results_df.columns:
                    column_order.append(column_name)
        
        # 添加提升幅度列
        improvement_cols = [col for col in results_df.columns if 'improvement' in col]
        column_order.extend(improvement_cols)
        
        # 重新排序列
        results_df = results_df.reindex(columns=[col for col in column_order if col in results_df.columns])
        
        # 保存到CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.csv")
        results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"详细结果已保存到: {results_file}")
        
        # 输出汇总统计
        logger.info("\n汇总统计:")
        
        # 各模型平均F1
        f1_columns = [col for col in results_df.columns if '_f1' in col]
        for col in f1_columns:
            model_name = col.replace('_f1', '')
            avg_f1 = results_df[col].mean()
            logger.info(f"{model_name} 平均F1: {avg_f1:.4f}")
        
        # 最佳模型分布
        if 'best_model' in results_df.columns:
            best_model_dist = results_df['best_model'].value_counts()
            logger.info("\n最佳模型分布:")
            for model, count in best_model_dist.items():
                percentage = count / len(results_df) * 100
                logger.info(f"  {model}: {count} 个方面 ({percentage:.1f}%)")
        
        # 计算平均提升
        if 'improvement_alce_vs_bert' in results_df.columns:
            avg_improvement = results_df['improvement_alce_vs_bert'].mean()
            logger.info(f"\nAnglE-ALCE 相比 BERT 平均提升: {avg_improvement:.4f}")
            
            # 统计提升为正的方面数
            positive_improvement = (results_df['improvement_alce_vs_bert'] > 0).sum()
            logger.info(f"AnglE-ALCE 优于 BERT 的方面数: {positive_improvement}/{len(results_df)}")
        
        if 'improvement_dot_vs_bert' in results_df.columns:
            avg_improvement = results_df['improvement_dot_vs_bert'].mean()
            logger.info(f"点积模型相比 BERT 平均提升: {avg_improvement:.4f}")
            
            # 统计提升为正的方面数
            positive_improvement = (results_df['improvement_dot_vs_bert'] > 0).sum()
            logger.info(f"点积模型优于 BERT 的方面数: {positive_improvement}/{len(results_df)}")
    
    # 输出最终总结
    print("\n" + "="*80)
    print("评估完成！")
    print(f"成功评估: {successful_aspects}/{len(ASPECT_MAPPING)} 个方面")
    print(f"详细日志和结果保存在: {RESULTS_DIR}")
    print("="*80)

def quick_evaluation_sample():
    """快速评估示例（用于测试）"""
    print("快速评估测试...")
    
    # 选择几个aspect进行快速测试
    test_aspects = {
        'Food#Taste': '口味',
        'Service#Hospitality': '服务态度',
        'Price#Cost_effective': '性价比'
    }
    
    logger = setup_logging()
    logger.info("快速评估测试 - 仅评估3个方面")
    
    all_results = []
    
    for aspect_en, aspect_cn in test_aspects.items():
        logger.info(f"\n评估: {aspect_en} ({aspect_cn})")
        result = evaluate_single_aspect(aspect_en, aspect_cn, logger)
        if result:
            all_results.append(result)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n快速评估结果:")
        print(results_df[['aspect_en', 'aspect_cn', 'bert_f1', 'dot_product_f1', 'angle_alce_f1']])

if __name__ == '__main__':
    # 检查路径和模型
    print("检查模型路径...")
    print(f"AnglE-ALCE 路径: {'✅ 存在' if os.path.exists(MODEL_PATH_ANGLE_ALCE) else '❌ 不存在'}")
    print(f"点积模型路径: {'✅ 存在' if os.path.exists(MODEL_PATH_DOT_PRODUCT) else '❌ 不存在'}")
    
    # 创建结果目录
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # 询问是否进行完整评估
    print("\n选择评估模式:")
    print("1. 完整评估所有18个方面 (耗时较长)")
    print("2. 快速评估测试 (仅评估3个方面)")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == '2':
        quick_evaluation_sample()
    else:
        # 完整评估
        evaluate_all_aspects()