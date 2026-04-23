import json
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)

class TripletDataConverter:
    """将 triplet 数据直接加载为 AnglE 兼容的三元组格式 (Format C)"""

    @staticmethod
    def load_and_convert(jsonl_file):
        """加载 JSONL 文件并保持三元组格式"""
        data = []
        valid_triplets = 0

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)

                    # 提取核心键，容错处理
                    query = item.get('query', '')
                    positive = item.get('positive', '')
                    negative = item.get('negative', '')

                    # 确保三者都不为空，才作为有效的三元组加入
                    if query and positive and negative:
                        data.append({
                            'query': query,
                            'positive': positive,
                            'negative': negative
                        })
                        valid_triplets += 1

        logger.info(f"📥 从文件读取并保留了 {valid_triplets} 条完整的 Triplet 数据 (Format C)")

        # 转换为 HuggingFace Dataset (直接传入字典列表，不进行任何拆分)
        dataset = Dataset.from_list(data)

        return dataset