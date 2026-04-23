import os
import json
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_lengths(tokenizer, texts, name=""):
    """统计文本列表的 token 长度"""
    lengths = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(tokens))
    lengths = np.array(lengths)
    logger.info(f"\n=== {name} 长度统计 ===")
    logger.info(f"样本数: {len(lengths)}")
    logger.info(f"最小长度: {np.min(lengths)}")
    logger.info(f"最大长度: {np.max(lengths)}")
    logger.info(f"平均长度: {np.mean(lengths):.1f}")
    logger.info(f"中位数: {np.median(lengths):.1f}")
    logger.info(f"90分位数: {np.percentile(lengths, 90):.1f}")
    logger.info(f"95分位数: {np.percentile(lengths, 95):.1f}")
    logger.info(f"99分位数: {np.percentile(lengths, 99):.1f}")
    return lengths

def plot_histogram(lengths, title, save_path=None):
    """绘制直方图"""
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=50, alpha=0.7, color='blue')
    plt.axvline(np.percentile(lengths, 95), color='red', linestyle='--', label='95th percentile')
    plt.axvline(np.mean(lengths), color='green', linestyle='--', label='mean')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"直方图已保存至: {save_path}")
    plt.show()

def main():
    # 设置路径（请根据实际情况修改）
    csv_file = "asap-master/data/train.csv"            # 原始 CSV 文件路径
    jsonl_file = "D:/WorkSpace/AnglE_yj/data_preparation/Aspect-Polarity_Pair/output/v2/asap_angle_contextual_ap_data_hybrid.jsonl"  # 动态窗口数据路径
    model_name = "hfl/chinese-roberta-wwm-ext"
    
    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 1. 统计原始 CSV 中的评论长度
    if os.path.exists(csv_file):
        reviews = []
        aspects = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                review = row.get('review', '').strip()
                if review:
                    reviews.append(review)
                # 收集方面名称（可选）
                for col in reader.fieldnames:
                    if col not in ['id', 'review', 'star']:
                        aspects.append(col.replace('#', ' '))
        
        # 统计完整评论长度
        if reviews:
            review_lengths = analyze_lengths(tokenizer, reviews, "原始评论（完整）")
            plot_histogram(review_lengths, "Raw Review Lengths (Full Text)", save_path="raw_review_lengths.png")
        # 统计方面名称长度（一般很短）
        if aspects:
            aspect_lengths = analyze_lengths(tokenizer, aspects, "方面名称")
            plot_histogram(aspect_lengths, "Aspect Name Lengths", save_path="aspect_name_lengths.png")
    else:
        logger.warning(f"CSV 文件不存在: {csv_file}")
    
    # 2. 统计动态窗口数据中的 text1（裁剪后的文本）长度
    if os.path.exists(jsonl_file):
        text1_list = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    text1 = item.get('text1', '')
                    if text1:
                        text1_list.append(text1)
                except:
                    continue
        if text1_list:
            window_lengths = analyze_lengths(tokenizer, text1_list, "动态窗口裁剪后文本")
            plot_histogram(window_lengths, "Windowed Text Lengths", save_path="windowed_text_lengths.png")
    else:
        logger.warning(f"JSONL 文件不存在: {jsonl_file}")

if __name__ == "__main__":
    main()