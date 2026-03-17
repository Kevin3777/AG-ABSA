import torch
import os
import sys
import numpy as np
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 使用相对路径 - 修复导入问题
current_dir = os.path.dirname(os.path.abspath(__file__))
angle_path = os.path.join(current_dir, '..', 'AnglE')
sys.path.insert(0, os.path.abspath(angle_path))

# 现在导入AnglE
from angle_emb import AnglE

# 手动实现余弦相似度函数（避免导入问题）
def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class AngleEncoder:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        try:
            angle = AnglE.from_pretrained(
                'hfl/chinese-roberta-wwm-ext',
                pretrained_model_path=model_path,
                pooling_strategy='cls'
            )
            
            if torch.cuda.is_available():
                angle = angle.cuda()
                logger.info("✅ AnglE编码器加载完成（使用GPU）")
            else:
                logger.info("✅ AnglE编码器加载完成（使用CPU）")
                
            return angle
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def encode_texts(self, texts):
        """编码文本列表"""
        try:
            return self.model.encode(texts, to_numpy=True)
        except Exception as e:
            logger.error(f"❌ 文本编码失败: {e}")
            return None
    
    def calculate_similarity(self, text1, text2):
        """计算两个文本的相似度"""
        try:
            vec1 = self.encode_texts([text1])[0]
            vec2 = self.encode_texts([text2])[0]
            similarity = cosine_similarity(vec1, vec2)
            return similarity
        except Exception as e:
            logger.error(f"❌ 相似度计算失败: {e}")
            return 0.0
    
    def find_most_similar(self, query, candidates):
        """在候选文本中查找最相似的"""
        try:
            query_vec = self.encode_texts([query])[0]
            candidate_vecs = self.encode_texts(candidates)
            
            similarities = []
            for i, cand_vec in enumerate(candidate_vecs):
                sim = cosine_similarity(query_vec, cand_vec)
                similarities.append((candidates[i], sim, i))
            
            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities
        except Exception as e:
            logger.error(f"❌ 最相似查找失败: {e}")
            return []
    
    def batch_similarity_matrix(self, texts):
        """计算文本之间的相似度矩阵"""
        try:
            vectors = self.encode_texts(texts)
            n = len(texts)
            similarity_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    similarity_matrix[i][j] = cosine_similarity(vectors[i], vectors[j])
            
            return similarity_matrix
        except Exception as e:
            logger.error(f"❌ 相似度矩阵计算失败: {e}")
            return None

def main():
    try:
        # 初始化编码器
        model_path = "checkpoints/angle_alce_encoder_final"
        logger.info(f"🚀 加载模型: {model_path}")
        encoder = AngleEncoder(model_path)
        
        # 示例1: 基本编码
        texts = [
            "这家餐厅的菜品味道很好",
            "食物美味，服务周到",
            "电脑运行速度很快",
            "酒店位置便利，靠近地铁站"
        ]
        
        logger.info("📝 文本编码示例:")
        vectors = encoder.encode_texts(texts)
        if vectors is not None:
            for i, text in enumerate(texts):
                logger.info(f"  '{text}'")
                logger.info(f"    向量维度: {vectors[i].shape}")
                logger.info(f"    向量范数: {np.linalg.norm(vectors[i]):.4f}")
        
        # 示例2: 相似度计算
        logger.info("\n🎯 相似度计算示例:")
        pairs = [
            ("产品质量很好", "商品质量不错"),
            ("服务质量差", "电脑运行速度快"),
            ("交通便利", "位置很方便"),
            ("今天天气不错", "今天天气很好")
        ]
        
        for text1, text2 in pairs:
            similarity = encoder.calculate_similarity(text1, text2)
            logger.info(f"  '{text1}'")
            logger.info(f"  '{text2}'")
            logger.info(f"  相似度: {similarity:.4f}")
            logger.info("  " + "="*40)
        
        # 示例3: 最相似查找
        logger.info("\n🔍 最相似查找示例:")
        query = "服务态度很好"
        candidates = [
            "员工服务热情",
            "产品质量优秀", 
            "服务员态度友好",
            "电脑性能强劲",
            "客服回应及时"
        ]
        
        results = encoder.find_most_similar(query, candidates)
        if results:
            logger.info(f"查询: '{query}'")
            logger.info("匹配结果:")
            for cand, sim, idx in results:
                logger.info(f"  相似度 {sim:.4f}: '{cand}'")
        
        # 示例4: 相似度矩阵
        logger.info("\n📊 相似度矩阵示例:")
        sample_texts = [
            "服务质量优秀",
            "服务态度很好", 
            "产品质量不错",
            "电脑运行快速"
        ]
        
        matrix = encoder.batch_similarity_matrix(sample_texts)
        if matrix is not None:
            logger.info("相似度矩阵:")
            for i, text in enumerate(sample_texts):
                row = "  ".join([f"{matrix[i][j]:.3f}" for j in range(len(sample_texts))])
                logger.info(f"  {text[:10]:<10} | {row}")
                
    except Exception as e:
        logger.error(f"💥 程序执行失败: {e}")

if __name__ == '__main__':
    main()