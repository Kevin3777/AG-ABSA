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

def setup_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = r'D:\WorkSpace\AnglE_yj'
    angle_root = os.path.join(project_root, 'AnglE')
    
    if os.path.exists(angle_root) and angle_root not in sys.path:
        sys.path.insert(0, angle_root)
        print(f"✅ 已添加 AnglE 根目录到路径: {angle_root}")
    elif angle_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"✅ 已添加项目根目录到路径: {project_root}")

setup_path()

# --- 导入 AnglE ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'AnglE', 'angle_emb'))
from angle_emb import AnglE

# --- 配置参数 ---
TEST_CSV_FILE = r'asap-master\data\test.csv' 
INDEX_DIR = 'rag_index'
MODEL_PATH = r'checkpoints\angle_alce_encoder_final\checkpoint-2605' 
TOP_K = 3 
LLM_MODEL = "THUDM/glm-4-9b-chat"
SILICONFLOW_API_KEY = "sk-mskwuczqlpvmmubgmtcejgolnnsapcumuyyuusgdgjanezfi"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# 评估 Aspect
ASPECT_TO_EVALUATE = 'Food#Taste' 
ASPECT_CN = '口味' 

# 日志文件路径
LOG_DIR = r"eval\12_5\log"
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------------------------------

def clean_review_text(text):
    text = str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\\', '', text) 
    text = text.replace('//', ' ')    
    return text.strip()

def load_test_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['review'] = df['review'].apply(clean_review_text)
        return df
    except Exception as e:
        print(f"❌ 错误：测试集加载失败，请检查路径。{e}")
        return None

class RAGPredictor:
    def __init__(self):
        print("正在加载资源...")
        self.device = 'cuda'
        
        # 1. 加载 AnglE
        self.angle = AnglE.from_pretrained(MODEL_PATH, pooling_strategy='cls').to(self.device)
        
        # 2. 加载索引
        self.index = faiss.read_index(os.path.join(INDEX_DIR, 'knowledge_base.index'))
        with open(os.path.join(INDEX_DIR, 'knowledge_base_meta.pkl'), 'rb') as f:
            self.kb_meta = pickle.load(f)
            
        print(f"知识库大小: {len(self.kb_meta)} 条记录")
        print("资源加载完成。")

    def retrieve(self, query_text, aspect_cn, k=TOP_K):
        """检索相似示例，返回详细信息"""
        angle_input = [f"Aspect: {aspect_cn}, Context: {query_text}"]
        query_vec = self.angle.encode(angle_input, to_numpy=True)
        faiss.normalize_L2(query_vec)
        
        # 搜索并获取相似度分数
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if 0 <= idx < len(self.kb_meta):
                meta = self.kb_meta[idx].copy()
                meta['similarity_score'] = float(1 - dist)  # 将距离转换为相似度
                meta['retrieval_rank'] = i + 1
                results.append(meta)
        return results

    def call_llm(self, review, aspect_cn, retrieved_examples, record_id):
        """调用LLM并记录完整信息"""
        # 构造 Few-Shot Prompt
        examples_str = ""
        polarity_map = {1: "正面", -1: "负面"} 
        
        for i, ex in enumerate(retrieved_examples):
            sentiment = polarity_map.get(ex['polarity'], "不适用")
            if sentiment != "不适用":
                examples_str += f"参考 {i+1}（相似度: {ex['similarity_score']:.3f}）：\"{ex['original_text']}\" -> 标签: {ex['polarity']}\n"
            
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
        
        # 记录发送的prompt
        prompt_log = {
            'record_id': record_id,
            'prompt_content': prompt,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1, 
            "max_tokens": 10 
        }
        
        try:
            response = requests.post(SILICONFLOW_API_URL, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                response_data = response.json()
                content = response_data['choices'][0]['message']['content'].strip()
                
                # 记录完整的API响应
                api_log = {
                    'record_id': record_id,
                    'api_response': content,
                    'full_response': json.dumps(response_data, ensure_ascii=False),
                    'status_code': response.status_code
                }
                
                # 尝试提取标签
                match = re.search(r'[+-]?\d+', content)
                if match:
                    val = int(match.group(0))
                    if val in [1, -1, 0]:
                        return val, prompt_log, api_log
            return 0, prompt_log, {}
        except Exception as e:
            print(f"API调用失败: {e}")
            return 0, {}, {}

def evaluate():
    df_test = load_test_data(TEST_CSV_FILE)
    if df_test is None: 
        return

    predictor = RAGPredictor()
    
    # 筛选出有明确标签的样本进行评估
    test_samples = df_test[df_test[ASPECT_TO_EVALUATE] != -2]
    test_samples = test_samples[test_samples[ASPECT_TO_EVALUATE].isin([1, -1])]
    
    # 如果样本太多，可以采样或限制数量
    if len(test_samples) > 1000:
        test_samples = test_samples.sample(n=1000, random_state=42)
        print(f"采样 1000 条样本进行评估...")
    
    y_true = []
    y_pred = []
    
    # 创建详细记录的数据结构
    detailed_records = []
    error_analysis = []
    prompt_logs = []
    api_logs = []
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(LOG_DIR, f"evaluation_details_{timestamp}.csv")
    error_file = os.path.join(LOG_DIR, f"error_analysis_{timestamp}.csv")
    
    print(f"\n评估 Aspect: {ASPECT_CN} ({ASPECT_TO_EVALUATE}) - 共 {len(test_samples)} 条明确情感样本...")
    
    for idx, (_, row) in tqdm(enumerate(test_samples.iterrows()), total=len(test_samples), desc="Evaluating Samples"):
        review = row['review']
        true_label = int(row[ASPECT_TO_EVALUATE])
        record_id = f"rec_{idx}_{timestamp}"
        
        # 1. 检索相似示例
        examples = predictor.retrieve(review, ASPECT_CN)
        
        # 2. 调用LLM进行预测
        pred_label, prompt_log, api_log = predictor.call_llm(review, ASPECT_CN, examples, record_id)
        
        # 3. 记录详细数据
        detailed_record = {
            'record_id': record_id,
            'review': review,
            'true_label': true_label,
            'pred_label': pred_label,
            'is_correct': 1 if true_label == pred_label else 0,
            'error_type': self._get_error_type(true_label, pred_label),
            'num_retrieved_examples': len(examples),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 添加检索到的示例信息
        for i, ex in enumerate(examples):
            detailed_record[f'retrieved_ex_{i+1}_text'] = ex.get('original_text', '')
            detailed_record[f'retrieved_ex_{i+1}_label'] = ex.get('polarity', '')
            detailed_record[f'retrieved_ex_{i+1}_similarity'] = ex.get('similarity_score', '')
            detailed_record[f'retrieved_ex_{i+1}_rank'] = ex.get('retrieval_rank', '')
        
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
                'error_type': self._get_error_type(true_label, pred_label),
                'retrieved_examples': json.dumps(examples, ensure_ascii=False),
                'prompt_snippet': prompt_log.get('prompt_content', '')[:200] + '...' if prompt_log else ''
            }
            error_analysis.append(error_record)
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        
        # 每隔50条保存一次中间结果
        if idx % 50 == 0 and idx > 0:
            temp_df = pd.DataFrame(detailed_records[-50:])
            temp_df.to_csv(os.path.join(LOG_DIR, f"temp_progress_{idx}.csv"), index=False, encoding='utf-8-sig')
        
        time.sleep(1)  # API限流

    # 保存所有详细记录
    if detailed_records:
        df_details = pd.DataFrame(detailed_records)
        df_details.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 详细记录已保存到: {output_file}")
    
    # 保存错误分析
    if error_analysis:
        df_errors = pd.DataFrame(error_analysis)
        df_errors.to_csv(error_file, index=False, encoding='utf-8-sig')
        print(f"✅ 错误分析已保存到: {error_file}")
    
    # 保存prompt日志
    if prompt_logs:
        df_prompts = pd.DataFrame(prompt_logs)
        df_prompts.to_csv(os.path.join(LOG_DIR, f"prompt_logs_{timestamp}.csv"), index=False, encoding='utf-8-sig')
    
    # 计算评估指标
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=[1, -1], average='macro')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # 生成分类报告
    target_names = ['负面(-1)', '正面(1)']
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # 保存评估报告
    report_data = {
        'aspect': ASPECT_TO_EVALUATE,
        'aspect_cn': ASPECT_CN,
        'total_samples': len(y_true),
        'accuracy': acc,
        'macro_f1': f1,
        'confusion_matrix': str(conf_matrix),
        'evaluation_time': timestamp,
        'model': LLM_MODEL,
        'top_k': TOP_K
    }
    
    # 添加详细指标
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                report_data[f"{label}_{metric}"] = value
    
    report_df = pd.DataFrame([report_data])
    report_df.to_csv(os.path.join(LOG_DIR, f"evaluation_report_{timestamp}.csv"), index=False, encoding='utf-8-sig')
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"=== RAG-P 评估结果 ({ASPECT_TO_EVALUATE}) ===")
    print(f"{'='*60}")
    print(f"评估样本数: {len(y_true)}")
    print(f"正确分类数: {sum(1 for t, p in zip(y_true, y_pred) if t == p)}")
    print(f"错误分类数: {len(y_true) - sum(1 for t, p in zip(y_true, y_pred) if t == p)}")
    print(f"错误率: {(len(y_true) - sum(1 for t, p in zip(y_true, y_pred) if t == p)) / len(y_true):.4f}")
    print(f"Accuracy (整体): {acc:.4f}")
    print(f"Macro F1 (正负情感): {f1:.4f}")
    print(f"\n混淆矩阵:")
    print(conf_matrix)
    print(f"\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    print(f"{'='*60}")
    
    # 打印一些错误分析的统计信息
    if error_analysis:
        print(f"\n📊 错误分析统计:")
        error_types = [e['error_type'] for e in error_analysis]
        from collections import Counter
        error_counts = Counter(error_types)
        for err_type, count in error_counts.items():
            print(f"  {err_type}: {count} 条 ({count/len(error_analysis)*100:.1f}%)")
        
        # 显示几个典型错误示例
        print(f"\n🔍 典型错误示例 (前3条):")
        for i, error in enumerate(error_analysis[:3]):
            print(f"\n示例 {i+1}:")
            print(f"  评论: {error['review'][:100]}...")
            print(f"  真实标签: {error['true_label']}, 预测标签: {error['pred_label']}")
            print(f"  错误类型: {error['error_type']}")

    def _get_error_type(self, true_label, pred_label):
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

# 将方法移到类外作为独立函数
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

# 修复：将get_error_type函数作为evaluate函数的内部函数
def evaluate():
    df_test = load_test_data(TEST_CSV_FILE)
    if df_test is None: 
        return

    predictor = RAGPredictor()
    
    # 筛选出有明确标签的样本进行评估
    test_samples = df_test[df_test[ASPECT_TO_EVALUATE] != -2]
    test_samples = test_samples[test_samples[ASPECT_TO_EVALUATE].isin([1, -1])]
    
    # 如果样本太多，可以采样或限制数量
    if len(test_samples) > 1000:
        test_samples = test_samples.sample(n=1000, random_state=42)
        print(f"采样 1000 条样本进行评估...")
    
    y_true = []
    y_pred = []
    
    # 创建详细记录的数据结构
    detailed_records = []
    error_analysis = []
    prompt_logs = []
    api_logs = []
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(LOG_DIR, f"evaluation_details_{timestamp}.csv")
    error_file = os.path.join(LOG_DIR, f"error_analysis_{timestamp}.csv")
    
    print(f"\n评估 Aspect: {ASPECT_CN} ({ASPECT_TO_EVALUATE}) - 共 {len(test_samples)} 条明确情感样本...")
    
    for idx, (_, row) in tqdm(enumerate(test_samples.iterrows()), total=len(test_samples), desc="Evaluating Samples"):
        review = row['review']
        true_label = int(row[ASPECT_TO_EVALUATE])
        record_id = f"rec_{idx}_{timestamp}"
        
        # 1. 检索相似示例
        examples = predictor.retrieve(review, ASPECT_CN)
        
        # 2. 调用LLM进行预测
        pred_label, prompt_log, api_log = predictor.call_llm(review, ASPECT_CN, examples, record_id)
        
        # 3. 记录详细数据
        detailed_record = {
            'record_id': record_id,
            'review': review,
            'true_label': true_label,
            'pred_label': pred_label,
            'is_correct': 1 if true_label == pred_label else 0,
            'error_type': get_error_type(true_label, pred_label),
            'num_retrieved_examples': len(examples),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 添加检索到的示例信息
        for i, ex in enumerate(examples):
            detailed_record[f'retrieved_ex_{i+1}_text'] = ex.get('original_text', '')
            detailed_record[f'retrieved_ex_{i+1}_label'] = ex.get('polarity', '')
            detailed_record[f'retrieved_ex_{i+1}_similarity'] = ex.get('similarity_score', '')
            detailed_record[f'retrieved_ex_{i+1}_rank'] = ex.get('retrieval_rank', '')
        
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
                'retrieved_examples': json.dumps(examples, ensure_ascii=False),
                'prompt_snippet': prompt_log.get('prompt_content', '')[:200] + '...' if prompt_log else ''
            }
            error_analysis.append(error_record)
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        
        # 每隔50条保存一次中间结果
        if idx % 50 == 0 and idx > 0:
            temp_df = pd.DataFrame(detailed_records[-50:])
            temp_df.to_csv(os.path.join(LOG_DIR, f"temp_progress_{idx}.csv"), index=False, encoding='utf-8-sig')
        
        time.sleep(1.5)  # API限流

    # 保存所有详细记录
    if detailed_records:
        df_details = pd.DataFrame(detailed_records)
        df_details.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 详细记录已保存到: {output_file}")
    
    # 保存错误分析
    if error_analysis:
        df_errors = pd.DataFrame(error_analysis)
        df_errors.to_csv(error_file, index=False, encoding='utf-8-sig')
        print(f"✅ 错误分析已保存到: {error_file}")
    
    # 保存prompt日志
    if prompt_logs:
        df_prompts = pd.DataFrame(prompt_logs)
        df_prompts.to_csv(os.path.join(LOG_DIR, f"prompt_logs_{timestamp}.csv"), index=False, encoding='utf-8-sig')
    
    # 计算评估指标
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=[1, -1], average='macro')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # 生成分类报告
    target_names = ['负面(-1)', '正面(1)']
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # 保存评估报告
    report_data = {
        'aspect': ASPECT_TO_EVALUATE,
        'aspect_cn': ASPECT_CN,
        'total_samples': len(y_true),
        'accuracy': acc,
        'macro_f1': f1,
        'confusion_matrix': str(conf_matrix),
        'evaluation_time': timestamp,
        'model': LLM_MODEL,
        'top_k': TOP_K
    }
    
    # 添加详细指标
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                report_data[f"{label}_{metric}"] = value
    
    report_df = pd.DataFrame([report_data])
    report_df.to_csv(os.path.join(LOG_DIR, f"evaluation_report_{timestamp}.csv"), index=False, encoding='utf-8-sig')
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"=== RAG-P 评估结果 ({ASPECT_TO_EVALUATE}) ===")
    print(f"{'='*60}")
    print(f"评估样本数: {len(y_true)}")
    print(f"正确分类数: {sum(1 for t, p in zip(y_true, y_pred) if t == p)}")
    print(f"错误分类数: {len(y_true) - sum(1 for t, p in zip(y_true, y_pred) if t == p)}")
    print(f"错误率: {(len(y_true) - sum(1 for t, p in zip(y_true, y_pred) if t == p)) / len(y_true):.4f}")
    print(f"Accuracy (整体): {acc:.4f}")
    print(f"Macro F1 (正负情感): {f1:.4f}")
    print(f"\n混淆矩阵:")
    print(conf_matrix)
    print(f"\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    print(f"{'='*60}")
    
    # 打印一些错误分析的统计信息
    if error_analysis:
        print(f"\n📊 错误分析统计:")
        error_types = [e['error_type'] for e in error_analysis]
        from collections import Counter
        error_counts = Counter(error_types)
        for err_type, count in error_counts.items():
            print(f"  {err_type}: {count} 条 ({count/len(error_analysis)*100:.1f}%)")
        
        # 显示几个典型错误示例
        print(f"\n🔍 典型错误示例 (前3条):")
        for i, error in enumerate(error_analysis[:3]):
            print(f"\n示例 {i+1}:")
            print(f"  评论: {error['review'][:100]}...")
            print(f"  真实标签: {error['true_label']}, 预测标签: {error['pred_label']}")
            print(f"  错误类型: {error['error_type']}")
            # 解析检索到的示例
            try:
                retrieved = json.loads(error['retrieved_examples'])
                print(f"  检索到的示例:")
                for j, ex in enumerate(retrieved[:2]):  # 只显示前2个
                    print(f"    示例{j+1}: {ex.get('original_text', '')[:80]}... -> 标签: {ex.get('polarity', '')}")
            except:
                pass

if __name__ == '__main__':
    evaluate()