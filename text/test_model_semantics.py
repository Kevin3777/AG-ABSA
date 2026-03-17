import sys
import os
import numpy as np
import torch
import io

# --- 解决Windows控制台编码问题 ---
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# --- 路径设置 ---
def setup_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    angle_root = os.path.join(project_root, 'AnglE')
    
    if os.path.exists(angle_root) and angle_root not in sys.path:
        sys.path.insert(0, angle_root)
        print(f"已添加 AnglE 根目录到路径: {angle_root}")
    elif angle_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"已添加项目根目录到路径: {project_root}")

setup_path()

from angle_emb import AnglE

# --- 配置参数 ---
MODEL_PATH = r'checkpoints\angle_alce_encoder_final\checkpoint-2605'

# 方面映射
ASPECT_MAPPING = {
    'Food#Taste': '口味',
    'Service#Hospitality': '服务态度',
    'Price#Level': '价格水平',
    'Ambience#Decoration': '装修装饰'
}

def create_ap_pair(aspect_chinese_name, review_context):
    """创建 AP Pair - 与训练时相同的格式"""
    return f"Aspect: {aspect_chinese_name}, Context: {review_context}"

def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def test_basic_semantics():
    """测试模型基本语义能力"""
    print("=" * 80)
    print("测试1: 模型基本语义能力")
    print("=" * 80)
    
    # 加载模型
    model_path = MODEL_PATH
    print(f"加载模型: {model_path}")
    
    try:
        angle = AnglE.from_pretrained(model_path, pooling_strategy='cls')
        if torch.cuda.is_available():
            angle = angle.cuda()
            print("✅ 模型已转移到GPU")
        else:
            print("⚠️ 使用CPU进行推理")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 测试相似的文本
    print("\n📊 测试相似文本的相似度:")
    
    # 组1: 咖啡相关的正面评论
    coffee_positive = [
        "这个咖啡味道很好，很香浓",
        "咖啡的味道很不错，香气浓郁",
        "拿铁很香，味道很好",
        "美式咖啡口感醇厚，很喜欢",
        "卡布奇诺的奶泡很细腻，味道不错"
    ]
    
    # 组2: 服务相关的负面评论
    service_negative = [
        "这家店的服务态度很差",
        "服务员很不耐烦，体验很糟糕",
        "服务人员态度恶劣，再也不来了",
        "店员爱理不理的，让人很不舒服",
        "服务太差了，等了半天也没人理"
    ]
    
    # 组3: 价格相关的评论
    price_mixed = [
        "价格太贵了，不值得",
        "性价比很高，很划算",
        "价格有点贵，但还能接受",
        "物美价廉，推荐",
        "价格适中，不算贵"
    ]
    
    # 测试组内相似度
    print("\n☕ 咖啡正面评论组内相似度:")
    coffee_vecs = angle.encode(coffee_positive, to_numpy=True)
    for i in range(len(coffee_positive)):
        for j in range(i+1, len(coffee_positive)):
            sim = cosine_similarity(coffee_vecs[i], coffee_vecs[j])
            print(f"  '{coffee_positive[i][:20]}...' vs '{coffee_positive[j][:20]}...': {sim:.4f}")
    
    print("\n😠 服务负面评论组内相似度:")
    service_vecs = angle.encode(service_negative, to_numpy=True)
    for i in range(len(service_negative)):
        for j in range(i+1, len(service_negative)):
            sim = cosine_similarity(service_vecs[i], service_vecs[j])
            print(f"  '{service_negative[i][:20]}...' vs '{service_negative[j][:20]}...': {sim:.4f}")
    
    # 测试跨组相似度
    print("\n🔄 咖啡正面 vs 服务负面（应较低）:")
    for i in range(min(3, len(coffee_positive))):
        for j in range(min(3, len(service_negative))):
            sim = cosine_similarity(coffee_vecs[i], service_vecs[j])
            print(f"  '{coffee_positive[i][:20]}...' vs '{service_negative[j][:20]}...': {sim:.4f}")

def test_format_impact():
    """测试格式化对相似度的影响"""
    print("\n" + "=" * 80)
    print("测试2: 格式化对相似度的影响")
    print("=" * 80)
    
    # 加载模型
    model_path = MODEL_PATH
    angle = AnglE.from_pretrained(model_path, pooling_strategy='cls')
    if torch.cuda.is_available():
        angle = angle.cuda()
    
    # 测试不同的格式
    test_texts = [
        "这个咖啡味道很好，很香浓",
        "披萨太难吃了，味道很差",
        "面条一般般，没什么特别的"
    ]
    
    formats = [
        ("原始文本", lambda t: t),
        ("Aspect格式", lambda t: create_ap_pair("口味", t)),
        ("简单格式", lambda t: f"方面：口味。{t}"),
    ]
    
    for format_name, format_func in formats:
        print(f"\n📝 格式: {format_name}")
        print("-" * 40)
        
        # 编码
        formatted_texts = [format_func(t) for t in test_texts]
        vectors = angle.encode(formatted_texts, to_numpy=True)
        
        # 显示格式示例
        print(f"  示例格式: {formatted_texts[0][:50]}...")
        
        # 计算相似度矩阵
        for i in range(len(test_texts)):
            for j in range(i+1, len(test_texts)):
                sim = cosine_similarity(vectors[i], vectors[j])
                print(f"  '{test_texts[i][:15]}...' vs '{test_texts[j][:15]}...': {sim:.4f}")
    
    # 测试相同文本不同格式的相似度
    print("\n🔄 相同文本不同格式的相似度:")
    text = "这个咖啡味道很好，很香浓"
    formats_to_test = [
        ("原始", text),
        ("Aspect", create_ap_pair("口味", text)),
        ("服务Aspect", create_ap_pair("服务态度", text)),  # 错误方面
        ("价格Aspect", create_ap_pair("价格水平", text)),  # 错误方面
    ]
    
    format_names = [f[0] for f in formats_to_test]
    format_texts = [f[1] for f in formats_to_test]
    
    vectors = angle.encode(format_texts, to_numpy=True)
    
    # 打印相似度矩阵
    print("  格式相似度矩阵:")
    print("  " + " " * 10 + " ".join([f"{name:12s}" for name in format_names]))
    
    for i in range(len(formats_to_test)):
        row = f"  {format_names[i]:10s}"
        for j in range(len(formats_to_test)):
            if i == j:
                row += " " * 13
            else:
                sim = cosine_similarity(vectors[i], vectors[j])
                row += f" {sim:11.4f}"
        print(row)

def test_vector_statistics():
    """测试向量统计特性"""
    print("\n" + "=" * 80)
    print("测试3: 向量统计特性")
    print("=" * 80)
    
    # 加载模型
    model_path = MODEL_PATH
    angle = AnglE.from_pretrained(model_path, pooling_strategy='cls')
    if torch.cuda.is_available():
        angle = angle.cuda()
    
    # 生成一些随机文本
    test_texts = []
    for i in range(100):
        # 生成简单文本
        if i < 33:
            test_texts.append(f"测试文本{i}，这是关于咖啡的正面评论，味道很好")
        elif i < 66:
            test_texts.append(f"测试文本{i}，这是关于服务的负面评论，态度很差")
        else:
            test_texts.append(f"测试文本{i}，这是关于价格的中性评论，不算贵")
    
    # 编码
    print("编码100个测试文本...")
    vectors = angle.encode(test_texts, to_numpy=True)
    
    # 计算统计信息
    print(f"\n📊 向量统计:")
    print(f"  向量形状: {vectors.shape}")
    print(f"  向量维度: {vectors.shape[1]}")
    
    # 计算均值、标准差
    mean_vec = np.mean(vectors, axis=0)
    std_vec = np.std(vectors, axis=0)
    
    print(f"  整体均值: {np.mean(mean_vec):.6f}")
    print(f"  整体标准差: {np.mean(std_vec):.6f}")
    
    # 计算向量范数
    norms = np.linalg.norm(vectors, axis=1)
    print(f"\n📏 向量范数统计:")
    print(f"  最小范数: {np.min(norms):.6f}")
    print(f"  最大范数: {np.max(norms):.6f}")
    print(f"  平均范数: {np.mean(norms):.6f}")
    print(f"  标准差: {np.std(norms):.6f}")
    
    # 检查异常值
    unusual_norms = norms[(norms < 0.5) | (norms > 2.0)]
    if len(unusual_norms) > 0:
        print(f"⚠️  发现 {len(unusual_norms)} 个异常范数:")
        for norm in unusual_norms[:5]:  # 只显示前5个
            print(f"    异常范数: {norm:.6f}")
    else:
        print("✅ 所有向量范数都在合理范围内")
    
    # 检查NaN和Inf
    if np.isnan(vectors).any():
        print("❌ 向量中包含NaN值！")
    else:
        print("✅ 向量中无NaN值")
    
    if np.isinf(vectors).any():
        print("❌ 向量中包含Inf值！")
    else:
        print("✅ 向量中无Inf值")

def test_faiss_compatibility():
    """测试与FAISS的兼容性"""
    print("\n" + "=" * 80)
    print("测试4: FAISS兼容性测试")
    print("=" * 80)
    
    import faiss
    
    # 加载模型
    model_path = MODEL_PATH
    angle = AnglE.from_pretrained(model_path, pooling_strategy='cls')
    if torch.cuda.is_available():
        angle = angle.cuda()
    
    # 生成测试向量
    test_texts = [
        "这个咖啡味道很好",
        "披萨太难吃了",
        "服务员态度很差",
        "价格有点贵但能接受",
        "环境很干净整洁"
    ]
    
    print("编码测试文本...")
    vectors = angle.encode(test_texts, to_numpy=True)
    
    # 原始向量统计
    print(f"\n原始向量:")
    print(f"  形状: {vectors.shape}")
    print(f"  范数范围: [{np.min(np.linalg.norm(vectors, axis=1)):.6f}, "
          f"{np.max(np.linalg.norm(vectors, axis=1)):.6f}]")
    
    # 归一化
    vectors_norm = vectors.copy()
    faiss.normalize_L2(vectors_norm)
    
    print(f"\n归一化后向量:")
    print(f"  范数范围: [{np.min(np.linalg.norm(vectors_norm, axis=1)):.6f}, "
          f"{np.max(np.linalg.norm(vectors_norm, axis=1)):.6f}]")
    
    # 创建FAISS索引
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 内积索引
    
    # 添加归一化后的向量
    index.add(vectors_norm)
    print(f"\nFAISS索引:")
    print(f"  向量数量: {index.ntotal}")
    
    # 测试查询
    query_text = "咖啡味道不错"
    query_vec = angle.encode([query_text], to_numpy=True)
    faiss.normalize_L2(query_vec)
    
    # 搜索
    k = 3
    distances, indices = index.search(query_vec, k)
    
    print(f"\n查询: '{query_text}'")
    print(f"FAISS搜索结果:")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if 0 <= idx < len(test_texts):
            print(f"  结果{i+1}: '{test_texts[idx][:30]}...' (相似度: {dist:.4f})")
    
    # 检查距离值范围
    print(f"\n距离值范围: [{np.min(distances):.4f}, {np.max(distances):.4f}]")
    if np.min(distances) < -1.0 or np.max(distances) > 1.0:
        print("⚠️  距离值超出[-1, 1]范围，可能有问题")
    else:
        print("✅ 距离值在合理范围内")

def test_aspect_awareness():
    """测试模型是否理解方面信息"""
    print("\n" + "=" * 80)
    print("测试5: 模型是否理解方面信息")
    print("=" * 80)
    
    # 加载模型
    model_path = MODEL_PATH
    angle = AnglE.from_pretrained(model_path, pooling_strategy='cls')
    if torch.cuda.is_available():
        angle = angle.cuda()
    
    # 测试用例：相同内容，不同方面
    test_cases = [
        {
            "content": "很好，很喜欢",
            "aspects": ["口味", "服务态度", "价格水平", "装修装饰"],
            "description": "通用正面评价"
        },
        {
            "content": "太差了，不喜欢",
            "aspects": ["口味", "服务态度", "价格水平", "装修装饰"],
            "description": "通用负面评价"
        },
        {
            "content": "味道很好，但是服务员态度很差",
            "aspects": ["口味", "服务态度"],
            "description": "混合评价"
        }
    ]
    
    for case in test_cases:
        print(f"\n📝 测试案例: {case['description']}")
        print(f"  内容: {case['content']}")
        
        # 编码不同方面的向量
        vectors = []
        for aspect in case['aspects']:
            formatted = create_ap_pair(aspect, case['content'])
            vec = angle.encode([formatted], to_numpy=True)[0]
            vectors.append((aspect, vec))
        
        # 计算相似度矩阵
        print("  方面间相似度:")
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                aspect_i, vec_i = vectors[i]
                aspect_j, vec_j = vectors[j]
                sim = cosine_similarity(vec_i, vec_j)
                print(f"    {aspect_i} vs {aspect_j}: {sim:.4f}")
        
        # 计算与原始文本的相似度
        original_vec = angle.encode([case['content']], to_numpy=True)[0]
        for aspect, vec in vectors:
            sim = cosine_similarity(original_vec, vec)
            print(f"    原始文本 vs {aspect}格式: {sim:.4f}")

def test_real_world_examples():
    """测试真实世界示例"""
    print("\n" + "=" * 80)
    print("测试6: 真实世界示例")
    print("=" * 80)
    
    # 加载模型
    model_path = MODEL_PATH
    angle = AnglE.from_pretrained(model_path, pooling_strategy='cls')
    if torch.cuda.is_available():
        angle = angle.cuda()
    
    # 实际评估中的查询和检索结果示例
    real_examples = [
        {
            "query": "第一次去他们家吃，位置很好找。环境很好，没有油烟。还可以自助小火锅，可以吃的菜也多。虾螃蟹鱿鱼味道都很不错。还有啤酒红酒果汁等饮料自助，选择特别多。果汁很好喝，尤其是黄色的哈密瓜汁。绿色的是黄瓜汁，我...",
            "retrieved": "平常大排档能吃到的这里都有虽说主打的是火锅但是炒菜也是一应俱全还特地点了一份我最爱的拔丝芋头味道还不错哦糟白菜肉末也很下饭二十元不到的蛋炒饭份量很足四个人吃都不用再另点米饭了铁板烧也是对得起它三十几元...",
            "similarity": 0.9873
        },
        {
            "query": "新宿舍第一次六人聚餐！心情很棒，这一餐也很好～因为人数比较多所以这次点的种类也多，董家湾的料种类向来非常全，分量也还不错！n店内环境宽敞整洁，每个人还有小围裙防止汤锅沸腾溅到衣服～n锅底点的鸳鸯锅68...",
            "retrieved": "看到人气美食介绍和好友们一起来拔草周日1点多到的人气依然等了10分钟左右四大1小点了招牌沙爹双拼由于女儿还小所以沙爹改为清汤锅麻辣锅里牛蛙很大还有龙利鱼味道不错辣的汗都出来了肥牛不错整块的涮到锅里不散...",
            "similarity": 0.9861
        }
    ]
    
    for i, example in enumerate(real_examples):
        print(f"\n📊 真实示例 {i+1}:")
        print(f"  查询: {example['query'][:80]}...")
        print(f"  检索到: {example['retrieved'][:80]}...")
        print(f"  记录相似度: {example['similarity']:.4f}")
        
        # 实际计算相似度
        query_vec = angle.encode([create_ap_pair("口味", example["query"])], to_numpy=True)[0]
        retrieved_vec = angle.encode([create_ap_pair("口味", example["retrieved"])], to_numpy=True)[0]
        
        actual_similarity = cosine_similarity(query_vec, retrieved_vec)
        print(f"  实际计算相似度: {actual_similarity:.4f}")
        
        if abs(actual_similarity - example["similarity"]) > 0.01:
            print(f"  ⚠️  差异较大: 差值 {abs(actual_similarity - example['similarity']):.4f}")
        else:
            print(f"  ✅ 匹配良好")

def main():
    """主函数"""
    print("🧪 AnglE模型语义测试")
    print("=" * 80)
    
    try:
        # 测试1: 基本语义能力
        test_basic_semantics()
        
        # 测试2: 格式化影响
        test_format_impact()
        
        # 测试3: 向量统计
        test_vector_statistics()
        
        # 测试4: FAISS兼容性
        test_faiss_compatibility()
        
        # 测试5: 方面理解
        test_aspect_awareness()
        
        # 测试6: 真实示例
        test_real_world_examples()
        
        print("\n" + "=" * 80)
        print("🎉 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()