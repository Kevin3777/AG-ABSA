# test_max_length_impact.py
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def test_max_length_impact():
    """测试max_length对评估结果的影响"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 模型路径
    model_path = r'checkpoints\angle_alce_encoder_final_V2\checkpoint-655'
    
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    
    # 创建简单测试数据
    texts = [
        "Aspect: 口味, Context: 这家餐厅的菜很好吃，味道正宗，服务也很周到，环境优雅",
        "Aspect: 口味, Context: 菜太难吃了，完全没有味道，价格还特别贵",
        "Aspect: 口味, Context: 味道一般，没什么特色，但也不难吃",
        "Aspect: 口味, Context: 非常美味，强烈推荐，是我吃过的最好吃的",
        "Aspect: 口味, Context: 太难吃了，绝对不会再来",
        "Aspect: 口味, Context: 还不错，可以接受",
        "Aspect: 口味, Context: 超级好吃，下次还会来",
        "Aspect: 口味, Context: 味道很差，服务也不好"
    ]
    labels = [1, -1, 0, 1, -1, 0, 1, -1]  # 1=正面, -1=负面, 0=中性
    
    # 过滤中性样本（只保留正面和负面）
    texts_filtered = [t for t, l in zip(texts, labels) if l != 0]
    labels_filtered = [l for l in labels if l != 0]
    
    print(f"测试样本数: {len(texts_filtered)} (正面: {labels_filtered.count(1)}, 负面: {labels_filtered.count(-1)})")
    
    # 测试不同的max_length
    max_lengths = [64, 128, 192, 256, 384, 512]
    
    results = {}
    
    for ml in max_lengths:
        print(f"\n{'='*60}")
        print(f"测试 max_length = {ml}")
        print('='*60)
        
        # 提取嵌入
        embeddings = []
        
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(
                texts_filtered,
                padding=True,
                truncation=True,
                max_length=ml,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**inputs)
            
            # 使用CLS向量
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings = cls_embeddings
        
        print(f"嵌入形状: {embeddings.shape}")
        print(f"嵌入范数: {np.linalg.norm(embeddings, axis=1)}")
        
        # 训练简单分类器
        clf = LogisticRegression(random_state=42)
        clf.fit(embeddings, labels_filtered)
        
        # 预测
        predictions = clf.predict(embeddings)
        
        # 计算指标
        accuracy = clf.score(embeddings, labels_filtered)
        f1 = f1_score(labels_filtered, predictions, average='macro', labels=[-1, 1])
        
        print(f"分类准确率: {accuracy:.4f}")
        print(f"Macro F1: {f1:.4f}")
        
        results[ml] = {
            'accuracy': accuracy,
            'f1': f1,
            'embeddings_norm': np.mean(np.linalg.norm(embeddings, axis=1))
        }
    
    # 分析结果
    print(f"\n{'='*60}")
    print("max_length影响分析")
    print('='*60)
    
    print("\n不同max_length的性能对比:")
    print("max_length | Accuracy | F1-Score | 嵌入范数")
    print("-" * 45)
    
    best_f1 = 0
    best_ml = 0
    
    for ml in max_lengths:
        acc = results[ml]['accuracy']
        f1 = results[ml]['f1']
        norm = results[ml]['embeddings_norm']
        print(f"{ml:10d} | {acc:8.4f} | {f1:8.4f} | {norm:8.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_ml = ml
    
    print(f"\n最佳max_length: {best_ml} (F1: {best_f1:.4f})")
    
    # 分析趋势
    print("\n趋势分析:")
    f1_values = [results[ml]['f1'] for ml in max_lengths]
    
    if max_lengths.index(192) < len(f1_values):
        f1_192 = f1_values[max_lengths.index(192)]
        print(f"训练时的max_length(192)对应的F1: {f1_192:.4f}")
    
    if max_lengths.index(512) < len(f1_values):
        f1_512 = f1_values[max_lengths.index(512)]
        print(f"评估时的max_length(512)对应的F1: {f1_512:.4f}")
        
        if 192 in max_lengths and 512 in max_lengths:
            diff = f1_512 - f1_192
            print(f"差异: {diff:.4f} ({'提升' if diff > 0 else '下降'})")

def test_token_length_distribution():
    """测试实际评论文本的token长度分布"""
    import pandas as pd
    
    print(f"\n{'='*60}")
    print("评论文本长度分析")
    print('='*60)
    
    # 加载一些实际数据
    try:
        train_file = r'asap-master\data\train.csv'
        df = pd.read_csv(train_file)
        
        # 清理文本
        def clean_text(text):
            import re
            text = str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            text = re.sub(r'\\', '', text) 
            text = text.replace('//', ' ')    
            return text.strip()
        
        df['review'] = df['review'].apply(clean_text)
        
        # 加载tokenizer
        model_path = r'checkpoints\angle_alce_encoder_final_V2\checkpoint-655'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 计算token长度
        token_lengths = []
        for text in df['review'].tolist()[:1000]:  # 抽样1000条
            tokens = tokenizer.encode(text, truncation=False)
            token_lengths.append(len(tokens))
        
        # 统计分析
        token_lengths = np.array(token_lengths)
        
        print(f"样本数: {len(token_lengths)}")
        print(f"平均token长度: {np.mean(token_lengths):.1f}")
        print(f"最小token长度: {np.min(token_lengths)}")
        print(f"最大token长度: {np.max(token_lengths)}")
        print(f"中位数token长度: {np.median(token_lengths):.1f}")
        
        # 长度分布
        print("\n长度分布:")
        thresholds = [64, 128, 192, 256, 384, 512]
        prev = 0
        for thresh in thresholds:
            count = ((token_lengths > prev) & (token_lengths <= thresh)).sum()
            percentage = count / len(token_lengths) * 100
            print(f"  {prev}-{thresh} tokens: {count}条 ({percentage:.1f}%)")
            prev = thresh
        
        # 超过512的
        over_512 = (token_lengths > 512).sum()
        if over_512 > 0:
            print(f"  超过512 tokens: {over_512}条 ({over_512/len(token_lengths)*100:.1f}%)")
        
        # 推荐max_length
        percentile_95 = np.percentile(token_lengths, 95)
        print(f"\n建议:")
        print(f"  95%的文本长度 ≤ {percentile_95:.1f} tokens")
        print(f"  为了覆盖大多数文本，建议max_length ≥ {int(np.ceil(percentile_95/64)*64)}")
        
    except Exception as e:
        print(f"数据分析失败: {e}")

if __name__ == '__main__':
    print("max_length对评估结果的影响测试")
    print("="*80)
    
    test_max_length_impact()
    test_token_length_distribution()