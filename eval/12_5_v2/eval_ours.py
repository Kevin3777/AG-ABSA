import pandas as pd
import faiss
import pickle
import numpy as np
import torch
import os
import sys
import requests
import json
import time
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import re
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import logging
from collections import Counter, defaultdict
import io

# --- 解决Windows控制台编码问题 ---
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# --- 配置日志 ---
def safe_logger():
    """创建安全的日志记录器"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger

logger = safe_logger()

# --- 路径设置 ---
def setup_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = r'D:\WorkSpace\AnglE_yj'
    angle_root = os.path.join(project_root, 'AnglE')
    
    if os.path.exists(angle_root) and angle_root not in sys.path:
        sys.path.insert(0, angle_root)
        logger.info(f"已添加 AnglE 根目录到路径: {angle_root}")
    elif angle_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.info(f"已添加项目根目录到路径: {project_root}")

setup_path()

from angle_emb import AnglE

# --- 配置参数 ---
TEST_CSV_FILE = r'asap-master\data\test.csv'
INDEX_DIR = 'rag_index_v2'
MODEL_PATH = r'checkpoints\angle_alce_encoder_final\checkpoint-2605'
TOP_K = 3
LLM_MODEL = "THUDM/glm-4-9b-chat"
SILICONFLOW_API_KEY = "sk-mskwuczqlpvmmubgmtcejgolnnsapcumuyyuusgdgjanezfi"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# 评估 Aspect
ASPECT_TO_EVALUATE = 'Food#Taste'
ASPECT_CN = '口味'

# 日志目录
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_DIR = f"eval\\12_5_v2\\log\\evaluation_logs_v2_{timestamp}"
os.makedirs(LOG_DIR, exist_ok=True)

# 方面映射（与构建脚本一致）
ASPECT_MAPPING = {
    'Location#Transportation': '交通', 'Location#Downtown': '地段', 'Location#Easy_to_find': '位置易找度',
    'Service#Queue': '排队', 'Service#Hospitality': '服务态度', 'Service#Parking': '停车', 'Service#Timely': '上菜速度',
    'Price#Level': '价格水平', 'Price#Cost_effective': '性价比', 'Price#Discount': '折扣优惠',
    'Ambience#Decoration': '装修装饰', 'Ambience#Noise': '噪音环境', 'Ambience#Space': '空间', 'Ambience#Sanitary': '卫生',
    'Food#Portion': '份量', 'Food#Taste': '口味', 'Food#Appearance': '外观卖相', 'Food#Recommend': '推荐度'
}

# ---------------------------------------------------------------------------------------------------

def clean_review_text(text):
    """清洗评论文本"""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\\', '', text)
    text = text.replace('//', ' ')
    text = re.sub(r'\s+', ' ', text)  # 合并多个空格
    return text.strip()

def load_test_data(file_path, aspect_to_evaluate, sample_size=None, random_state=42):
    """加载测试数据"""
    logger.info(f"加载测试数据: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"加载成功，共 {len(df)} 条数据")
        
        # 清洗文本
        df['review'] = df['review'].apply(clean_review_text)
        
        # 筛选指定方面
        df = df[df[aspect_to_evaluate] != -2]  # 排除不相关样本
        
        # 只保留明确情感标签
        df = df[df[aspect_to_evaluate].isin([1, -1])]
        
        logger.info(f"筛选后剩余 {len(df)} 条明确情感样本")
        
        # 采样
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=random_state)
            logger.info(f"采样 {sample_size} 条样本进行评估")
        
        return df
    
    except Exception as e:
        logger.error(f"测试集加载失败: {e}")
        return None

class PracticalRAGPredictor:
    """实用版RAG预测器"""
    
    def __init__(self, index_dir=INDEX_DIR, model_path=MODEL_PATH):
        logger.info("正在加载RAG预测器资源...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. 加载 AnglE
        try:
            self.angle = AnglE.from_pretrained(model_path, pooling_strategy='cls').to(self.device)
            logger.info(f"模型加载成功: {model_path}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
        
        # 2. 加载索引和元数据
        index_path = os.path.join(index_dir, 'knowledge_base.index')
        meta_path = os.path.join(index_dir, 'knowledge_base_meta.pkl')
        
        if not os.path.exists(index_path):
            logger.error(f"索引文件不存在: {index_path}")
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
        
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            self.kb_meta = pickle.load(f)
        
        logger.info(f"索引信息: {self.index.ntotal} 个向量")
        logger.info("RAG预测器初始化完成")
    
    def retrieve(self, query_text, aspect_cn, k=TOP_K):
        """
        检索相似示例
        """
        # 构建查询文本
        angle_input = [f"Aspect: {aspect_cn}, Context: {query_text}"]
        
        # 编码查询
        query_vec = self.angle.encode(angle_input, to_numpy=True)
        faiss.normalize_L2(query_vec)
        
        # 搜索
        distances, indices = self.index.search(query_vec, k)
        
        # 收集结果
        results = []
        
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.kb_meta):
                similarity = float(dist)  # 内积索引，dist就是相似度
                meta = self.kb_meta[idx].copy()
                meta['similarity_score'] = similarity
                meta['faiss_distance'] = float(dist)
                results.append(meta)
        
        # 按相似度排序
        results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)[:k]
        
        return results
    
    def call_llm(self, review, aspect_cn, retrieved_examples, record_id=None):
        """调用LLM进行预测（使用你指定的Prompt格式）"""
        if not retrieved_examples:
            logger.warning(f"没有检索到示例，将返回中性标签")
            return 0, {}, {}
        
        # 构造 Few-Shot Prompt（按照你指定的格式）
        examples_str = ""
        
        for i, ex in enumerate(retrieved_examples[:3]):  # 最多3个示例
            examples_str += f"参考 {i+1}：\"{ex['original_text']}\" -> 标签: {ex['polarity']}\n"
        
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
        
        # 记录prompt
        prompt_log = {
            'record_id': record_id,
            'prompt_content': prompt,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 20
        }
        
        try:
            response = requests.post(SILICONFLOW_API_URL, json=payload, headers=headers, timeout=15)
            
            api_log = {
                'record_id': record_id,
                'status_code': response.status_code,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if response.status_code == 1000:
                response_data = response.json()
                content = response_data['choices'][0]['message']['content'].strip()
                api_log['api_response'] = content
                
                # 提取标签
                match = re.search(r'\[标签:\s*([+-]?\d+)\]', content)
                if not match:
                    match = re.search(r'标签:\s*([+-]?\d+)', content)
                if not match:
                    match = re.search(r'([+-]?\d+)', content)
                
                if match:
                    val = int(match.group(1))
                    if val in [1, -1, 0]:
                        return val, prompt_log, api_log
                    else:
                        logger.warning(f"提取到无效标签: {val}")
                
                # 如果未提取到数字，尝试从文本判断
                if '正面' in content or '1' in content:
                    return 1, prompt_log, api_log
                elif '负面' in content or '-1' in content:
                    return -1, prompt_log, api_log
            
            return 0, prompt_log, api_log
            
        except requests.exceptions.Timeout:
            logger.error(f"API请求超时")
            return 0, prompt_log, {'error': 'timeout'}
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            return 0, prompt_log, {'error': str(e)}

def evaluate_aspect(df_test, aspect_to_evaluate, aspect_cn, sample_size=1000, api_delay=1.5):
    """评估特定方面"""
    logger.info(f"\n{'='*60}")
    logger.info(f"开始评估方面: {aspect_cn} ({aspect_to_evaluate})")
    logger.info(f"{'='*60}")
    
    # 初始化预测器
    try:
        predictor = PracticalRAGPredictor()
    except Exception as e:
        logger.error(f"RAG预测器初始化失败: {e}")
        return None
    
    # 准备样本
    test_samples = df_test[df_test[aspect_to_evaluate].isin([1, -1])]
    
    if sample_size and sample_size < len(test_samples):
        test_samples = test_samples.sample(n=sample_size, random_state=42)
    
    logger.info(f"评估样本数: {len(test_samples)}")
    
    # 初始化记录
    detailed_records = []
    error_analysis = []
    prompt_logs = []
    api_logs = []
    
    y_true = []
    y_pred = []
    
    # 进度条
    pbar = tqdm(test_samples.iterrows(), total=len(test_samples), desc="评估进度")
    
    for idx, (_, row) in enumerate(pbar):
        review = row['review']
        true_label = int(row[aspect_to_evaluate])
        record_id = f"rec_{idx}_{timestamp}"
        
        # 1. 检索相似示例
        examples = predictor.retrieve(review, aspect_cn, k=TOP_K)
        
        # 2. 调用LLM进行预测
        pred_label, prompt_log, api_log = predictor.call_llm(
            review, aspect_cn, examples, record_id
        )
        
        # 3. 记录详细数据
        detailed_record = {
            'record_id': record_id,
            'review': review,
            'true_label': true_label,
            'pred_label': pred_label,
            'is_correct': 1 if true_label == pred_label else 0,
            'error_type': get_error_type(true_label, pred_label),
            'num_retrieved_examples': len(examples),
            'avg_similarity': np.mean([ex.get('similarity_score', 0) for ex in examples]) if examples else 0,
            'max_similarity': max([ex.get('similarity_score', 0) for ex in examples]) if examples else 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 添加检索到的示例信息
        for i, ex in enumerate(examples):
            detailed_record[f'retrieved_ex_{i+1}_text'] = ex['original_text'][:200] + ('...' if len(ex['original_text']) > 200 else '')
            detailed_record[f'retrieved_ex_{i+1}_label'] = ex['polarity']
            detailed_record[f'retrieved_ex_{i+1}_similarity'] = ex.get('similarity_score', 0)
        
        detailed_records.append(detailed_record)
        
        # 记录prompt和API日志
        if prompt_log:
            prompt_logs.append(prompt_log)
        if api_log:
            api_logs.append(api_log)
        
        # 如果是错误分类，记录到错误分析
        if true_label != pred_label:
            error_record = {
                'record_id': record_id,
                'review': review,
                'true_label': true_label,
                'pred_label': pred_label,
                'error_type': get_error_type(true_label, pred_label),
                'avg_similarity': detailed_record['avg_similarity'],
                'max_similarity': detailed_record['max_similarity'],
                'retrieved_examples_count': len(examples),
                'retrieved_labels': [ex['polarity'] for ex in examples]
            }
            error_analysis.append(error_record)
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        
        # 更新进度条描述
        correct_count = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct_count / len(y_true) if y_true else 0
        pbar.set_postfix({'准确率': f'{accuracy:.3f}', '正确数': correct_count})
        
        # API限流
        if api_delay > 0:
            time.sleep(api_delay)
    
    # 保存详细记录
    if detailed_records:
        df_details = pd.DataFrame(detailed_records)
        details_file = os.path.join(LOG_DIR, f'evaluation_details_{aspect_to_evaluate}.csv')
        df_details.to_csv(details_file, index=False, encoding='utf-8-sig')
        logger.info(f"详细记录已保存到: {details_file}")
    
    # 保存错误分析
    if error_analysis:
        df_errors = pd.DataFrame(error_analysis)
        errors_file = os.path.join(LOG_DIR, f'error_analysis_{aspect_to_evaluate}.csv')
        df_errors.to_csv(errors_file, index=False, encoding='utf-8-sig')
        logger.info(f"错误分析已保存到: {errors_file}")
    
    # 保存prompt日志
    if prompt_logs:
        df_prompts = pd.DataFrame(prompt_logs)
        prompts_file = os.path.join(LOG_DIR, f'prompt_logs_{aspect_to_evaluate}.csv')
        df_prompts.to_csv(prompts_file, index=False, encoding='utf-8-sig')
        logger.info(f"Prompt日志已保存到: {prompts_file}")
    
    # 计算评估指标
    if len(y_true) == 0:
        logger.error("没有评估结果")
        return None
    
    return calculate_metrics(y_true, y_pred, aspect_to_evaluate, aspect_cn, detailed_records)

def calculate_metrics(y_true, y_pred, aspect_to_evaluate, aspect_cn, detailed_records):
    """计算评估指标"""
    # 基础指标
    acc = accuracy_score(y_true, y_pred)
    
    # 只对正负情感计算F1
    labels_to_eval = [1, -1]
    y_true_filtered = [t for t, p in zip(y_true, y_pred) if t in labels_to_eval and p in labels_to_eval]
    y_pred_filtered = [p for t, p in zip(y_true, y_pred) if t in labels_to_eval and p in labels_to_eval]
    
    if len(y_true_filtered) > 0:
        macro_f1 = f1_score(y_true_filtered, y_pred_filtered, average='macro')
    else:
        macro_f1 = 0
    
    # 混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
    
    # 分类报告
    target_names = ['负面(-1)', '中性(0)', '正面(1)']
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # 检索质量统计
    avg_similarities = [r['avg_similarity'] for r in detailed_records]
    max_similarities = [r['max_similarity'] for r in detailed_records]
    
    retrieval_stats = {
        'avg_similarity_mean': np.mean(avg_similarities),
        'avg_similarity_std': np.std(avg_similarities),
        'max_similarity_mean': np.mean(max_similarities),
        'max_similarity_std': np.std(max_similarities),
        'low_similarity_count': sum(1 for s in max_similarities if s < 0.3),
        'medium_similarity_count': sum(1 for s in max_similarities if 0.3 <= s < 0.6),
        'high_similarity_count': sum(1 for s in max_similarities if s >= 0.6)
    }
    
    # 保存评估报告
    report_data = {
        'aspect': aspect_to_evaluate,
        'aspect_cn': aspect_cn,
        'total_samples': len(y_true),
        'accuracy': acc,
        'macro_f1': macro_f1,
        'confusion_matrix': str(conf_matrix.tolist()),
        'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': LLM_MODEL,
        'top_k': TOP_K,
        'api_delay': 1.5,
        'retrieval_stats': retrieval_stats
    }
    
    # 添加详细指标
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                report_data[f"{label}_{metric}"] = value
    
    report_file = os.path.join(LOG_DIR, f'evaluation_report_{aspect_to_evaluate}.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    logger.info(f"\n{'='*60}")
    logger.info(f"=== RAG-P 评估结果 ({aspect_cn}) ===")
    logger.info(f"{'='*60}")
    logger.info(f"评估样本数: {len(y_true)}")
    logger.info(f"正确分类数: {sum(1 for t, p in zip(y_true, y_pred) if t == p)}")
    logger.info(f"错误分类数: {len(y_true) - sum(1 for t, p in zip(y_true, y_pred) if t == p)}")
    logger.info(f"Accuracy (整体): {acc:.4f}")
    logger.info(f"Macro F1 (正负情感): {macro_f1:.4f}")
    
    logger.info(f"\n检索质量统计:")
    logger.info(f"  平均相似度: {retrieval_stats['avg_similarity_mean']:.4f} ± {retrieval_stats['avg_similarity_std']:.4f}")
    logger.info(f"  最高相似度: {retrieval_stats['max_similarity_mean']:.4f} ± {retrieval_stats['max_similarity_std']:.4f}")
    logger.info(f"  高相似度(≥0.6): {retrieval_stats['high_similarity_count']} 条")
    logger.info(f"  中相似度(0.3-0.6): {retrieval_stats['medium_similarity_count']} 条")
    logger.info(f"  低相似度(<0.3): {retrieval_stats['low_similarity_count']} 条")
    
    logger.info(f"\n混淆矩阵:")
    logger.info(f"       预测")
    logger.info(f"       -1   0   1")
    logger.info(f"实  -1  {conf_matrix[0][0]:4d} {conf_matrix[0][1]:4d} {conf_matrix[0][2]:4d}")
    logger.info(f"际   0  {conf_matrix[1][0]:4d} {conf_matrix[1][1]:4d} {conf_matrix[1][2]:4d}")
    logger.info(f"      1  {conf_matrix[2][0]:4d} {conf_matrix[2][1]:4d} {conf_matrix[2][2]:4d}")
    
    # 错误类型分析
    if len(y_true) > sum(1 for t, p in zip(y_true, y_pred) if t == p):
        error_types = [get_error_type(t, p) for t, p in zip(y_true, y_pred) if t != p]
        error_counts = Counter(error_types)
        
        logger.info(f"\n错误类型分析:")
        for err_type, count in error_counts.most_common():
            percentage = count / len(error_types) * 100
            logger.info(f"  {err_type}: {count} 条 ({percentage:.1f}%)")
    
    logger.info(f"\n评估报告已保存到: {report_file}")
    
    return report_data

def get_error_type(true_label, pred_label):
    """获取错误类型描述"""
    if true_label == pred_label:
        return "正确"
    elif pred_label == 0 and true_label != 0:
        return "预测为中性(漏报)"
    elif true_label == 0 and pred_label != 0:
        return "中性预测为情感(误报)"
    elif true_label == 1 and pred_label == -1:
        return "正面误判为负面"
    elif true_label == -1 and pred_label == 1:
        return "负面误判为正面"
    else:
        return "其他错误"

def main():
    """主评估函数"""
    print("RAG系统评估程序")
    print("-" * 50)
    
    # 检查必要文件
    if not os.path.exists(INDEX_DIR):
        print(f"错误: 索引目录不存在: {INDEX_DIR}")
        print("请先运行 build_rag_index_v2.py 构建知识库")
        return
    
    # 加载测试数据
    df_test = load_test_data(TEST_CSV_FILE, ASPECT_TO_EVALUATE, sample_size=1000)
    
    if df_test is None or len(df_test) == 0:
        print("测试数据加载失败或为空")
        return
    
    print(f"\n测试数据统计:")
    print(f"  总样本数: {len(df_test)}")
    print(f"  方面: {ASPECT_CN} ({ASPECT_TO_EVALUATE})")
    
    label_counts = df_test[ASPECT_TO_EVALUATE].value_counts()
    print(f"  标签分布:")
    for label, count in label_counts.items():
        sentiment = "正面" if label == 1 else "负面"
        print(f"    {sentiment}({label}): {count} 条")
    
    # 评估
    print(f"\n开始评估...")
    print(f"  日志目录: {LOG_DIR}")
    print(f"  API模型: {LLM_MODEL}")
    print(f"  检索数量: Top-{TOP_K}")
    
    result = evaluate_aspect(
        df_test=df_test,
        aspect_to_evaluate=ASPECT_TO_EVALUATE,
        aspect_cn=ASPECT_CN,
        sample_size=1000,
        api_delay=1.5
    )
    
    if result:
        print(f"\n评估完成!")
        print(f"  准确率: {result['accuracy']:.4f}")
        print(f"  Macro F1: {result['macro_f1']:.4f}")
        print(f"  详细结果请查看: {LOG_DIR}/")
    else:
        print(f"\n评估失败")

if __name__ == '__main__':
    main()