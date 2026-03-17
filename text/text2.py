# test_original_angle.py
import os
import sys
import numpy as np

# 添加AnglE库到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = r'D:\WorkSpace\AnglE_yj'
angle_root = os.path.join(project_root, 'AnglE')

if os.path.exists(angle_root) and angle_root not in sys.path:
    sys.path.insert(0, angle_root)
    print(f"✅ 已添加AnglE根目录到路径: {angle_root}")

try:
    from angle_emb import AnglE
    import torch
    
    print("✅ 成功导入AnglE库")
    
    # 测试模型
    model_path = r'checkpoints\angle_alce_encoder_final_V2\checkpoint-655'
    print(f"测试模型: {model_path}")
    
    # 创建测试文本
    test_texts = [
        "Aspect: 口味, Context: 这家餐厅的菜很好吃，味道正宗",
        "Aspect: 口味, Context: 菜太难吃了，完全没有味道",
    ]
    
    # 测试不同的参数组合
    test_cases = [
        {'pooling_strategy': 'cls', 'normalize_embedding': False, 'max_length': 192},
        {'pooling_strategy': 'cls', 'normalize_embedding': True, 'max_length': 192},
        {'pooling_strategy': 'mean', 'normalize_embedding': False, 'max_length': 192},
        {'pooling_strategy': 'mean', 'normalize_embedding': True, 'max_length': 192},
        {'pooling_strategy': 'cls', 'normalize_embedding': False, 'max_length': 512},
    ]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for i, params in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"测试用例 {i+1}: {params}")
        print('='*60)
        
        try:
            model = AnglE.from_pretrained(
                model_path,
                pooling_strategy=params['pooling_strategy'],
                train_mode=False
            ).to(device)
            
            embeddings = model.encode(
                test_texts,
                batch_size=2,
                to_numpy=True,
                device=device,
                normalize_embedding=params['normalize_embedding'],
                padding='longest',
                max_length=params['max_length']
            )
            
            print(f"嵌入形状: {embeddings.shape}")
            print(f"嵌入范数: {np.linalg.norm(embeddings, axis=1)}")
            print(f"嵌入均值: {np.mean(embeddings):.6f}")
            print(f"嵌入标准差: {np.std(embeddings):.6f}")
            
            # 计算两个嵌入的余弦相似度
            if embeddings.shape[0] >= 2:
                cos_sim = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
                print(f"余弦相似度: {cos_sim:.6f}")
                
        except Exception as e:
            print(f"测试失败: {e}")
    
except ImportError as e:
    print(f"❌ 无法导入AnglE: {e}")
    print("请确保在正确环境中运行")
except Exception as e:
    print(f"❌ 发生错误: {e}")
    import traceback
    traceback.print_exc()