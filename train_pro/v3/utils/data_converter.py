import json
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)

class TripletDataConverter:
    """将 triplet 数据转换为 AnglE 兼容的格式"""
    
    @staticmethod
    def convert_triplet_to_pairs(triplet_data):
        """
        将 triplet 数据转换为文本对格式
        格式: {"query": "...", "positive": "...", "negative": "...", "type": "..."}
        """
        pairs_data = []
        
        for item in triplet_data:
            query = item.get('query', '')
            positive = item.get('positive', '')
            negative = item.get('negative', '')
            
            # 添加正样本对 (query, positive) -> label=1
            if query and positive:
                pairs_data.append({
                    'text1': query,
                    'text2': positive,
                    'label': 1
                })
            
            # 添加负样本对 (query, negative) -> label=0
            if query and negative:
                pairs_data.append({
                    'text1': query,
                    'text2': negative,
                    'label': 0
                })
        
        return pairs_data
    
    @staticmethod
    def load_and_convert(jsonl_file):
        """加载 JSONL 文件并转换为 AnglE 格式"""
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        logger.info(f"📥 加载了 {len(data)} 条 triplet 数据")
        
        # 转换为文本对格式
        pairs_data = TripletDataConverter.convert_triplet_to_pairs(data)
        
        logger.info(f"🔄 转换为 {len(pairs_data)} 条文本对数据")
        logger.info(f"📊 正样本: {len([x for x in pairs_data if x['label'] == 1])}")
        logger.info(f"📊 负样本: {len([x for x in pairs_data if x['label'] == 0])}")
        
        # 转换为 HuggingFace Dataset
        dataset = Dataset.from_list(pairs_data)
        
        return dataset