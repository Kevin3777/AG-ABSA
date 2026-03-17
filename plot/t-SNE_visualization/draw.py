import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import os

# 添加模块路径到系统路径
sys.path.append('train_learnable/v3_all1')
from train_encoder import AnglE  # 导入你写好的模型类

# Aspect 映射
ASPECT_MAPPING = {
    'Location#Transportation': '交通',
    'Location#Downtown': '地段',
    'Location#Easy_to_find': '位置易找度',
    'Service#Queue': '排队',
    'Service#Parking': '停车',
    'Service#Timely': '上菜速度',
    'Price#Level': '价格水平',
    'Price#Discount': '折扣优惠',
    'Ambience#Decoration': '装修装饰',
    'Ambience#Noise': '噪音环境',
    'Ambience#Space': '空间',
    'Ambience#Sanitary': '卫生',
    'Food#Portion': '份量',
    'Food#Appearance': '外观卖相',
    'Service#Hospitality': '服务态度',
    'Price#Cost_effective': '性价比',
    'Food#Taste': '口味',
    'Food#Recommend': '推荐度'
}

def load_real_data(file_path, selected_aspects=None, max_samples_per_class=30, verbose=False):
    """
    加载真实数据
    
    Args:
        file_path: CSV文件路径
        selected_aspects: 要选择的Aspect列表
        max_samples_per_class: 每个类别最多采样多少个样本
        verbose: 是否打印详细信息
    """
    if selected_aspects is None:
        selected_aspects = ['Ambience#Decoration', 'Service#Hospitality', 'Food#Taste']
    
    texts = []
    aspects = []
    polarities = []
    
    # 用于按类别收集样本
    class_samples = {}
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        text_col = 'review'
        
        # 检查aspect列
        existing_columns = [a for a in selected_aspects if a in df.columns]
        
        # 先按类别收集所有样本
        for idx, row in df.iterrows():
            text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            if not text.strip():
                continue
            
            for aspect in existing_columns:
                if pd.notna(row[aspect]):
                    polarity_val = row[aspect]
                    
                    polarity = None
                    if isinstance(polarity_val, (int, float)):
                        if polarity_val > 0:
                            polarity = 'Positive'
                        elif polarity_val < 0:
                            polarity = 'Negative'
                    
                    if polarity:
                        key = f"{aspect}_{polarity}"
                        if key not in class_samples:
                            class_samples[key] = []
                        class_samples[key].append((text, aspect, polarity))
                        break
        
        # ⚠️ 在这里采样：每个类别只取 max_samples_per_class 个
        for key, samples in class_samples.items():
            # 如果样本数超过限制，随机采样
            if len(samples) > max_samples_per_class:
                import random
                sampled = random.sample(samples, max_samples_per_class)
            else:
                sampled = samples
            
            # 添加到最终结果
            for text, aspect, polarity in sampled:
                texts.append(text)
                aspects.append(aspect)
                polarities.append(polarity)
        
        if verbose:
            print(f"采样后共 {len(texts)} 条数据")
            for key, samples in class_samples.items():
                print(f"  {key}: 原始{len(samples)}条 -> 采样{min(len(samples), max_samples_per_class)}条")
        
    except Exception as e:
        print(f"读取数据出错: {e}")
        return [], [], []
    
    return texts, aspects, polarities

def get_embeddings(model_path, texts, batch_size=32):
    """批量提取特征"""
    print(f"🧠 正在提取特征: {model_path} ...")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return None
    
    try:
        model = AnglE.from_pretrained(model_path, train_mode=False)
        if torch.cuda.is_available():
            model.cuda()
            print("  使用GPU加速")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None
    
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = []
        
        with torch.no_grad():
            for text in batch_texts:
                try:
                    vec = model.encode(text, to_numpy=True)
                    # 确保输出是1D向量
                    if len(vec.shape) > 1:
                        vec = vec[0]
                    batch_embeddings.append(vec)
                except Exception as e:
                    print(f"  处理文本时出错: {text[:50]}...")
                    # 使用随机向量作为fallback（不推荐，但避免程序崩溃）
                    batch_embeddings.append(np.random.randn(768))
            
        embeddings.extend(batch_embeddings)
        
        # 打印进度
        progress = (i + len(batch_texts)) / len(texts) * 100
        print(f"  进度: {i+len(batch_texts)}/{len(texts)} ({progress:.1f}%)")
    
    return np.array(embeddings)

def plot_tsne_comparison(selected_aspects=None, output_prefix='tsne'):
    """
    绘制t-SNE对比图
    
    Args:
        selected_aspects: 要分析的aspect列表
        output_prefix: 输出文件前缀
    """
    # 文件路径
    test_data_path = "asap-master/data/test.csv"
    
    # 替换为你那个 F1=0.8756 (无 Mask / Format C) 的模型路径
    path_baseline = "checkpoints_standard_angle/v1" 
    # 替换为你那个 F1=0.8851 (SOTA) 的模型路径
    path_ours = "checkpoints_learnable/v3_all1"

    # 1. 加载数据
    # all_texts, labels_aspect, labels_polarity = load_real_data(test_data_path, selected_aspects)
    
    all_texts, labels_aspect, labels_polarity = load_real_data(
    test_data_path,
    selected_aspects,
    max_samples_per_class=30,  # ⚠️ 添加这行，可以调整这个数字
    verbose=True
)
    
    
    

    if len(all_texts) < 10:
        print("❌ 数据量太少，无法进行有意义的t-SNE分析")
        # 显示一些示例数据帮助调试
        print("\n调试信息: 请检查CSV文件中的极性值格式")
        try:
            df_sample = pd.read_csv(test_data_path, encoding='utf-8-sig').head(5)
            for aspect in selected_aspects:
                if aspect in df_sample.columns:
                    print(f"\n{aspect} 列的前5个值:")
                    print(df_sample[aspect].tolist())
        except:
            pass
        return
    
    # 2. 提取特征
    print(f"\n开始提取特征，共 {len(all_texts)} 条数据...")
    
    emb_baseline = get_embeddings(path_baseline, all_texts)
    if emb_baseline is None:
        print("❌ Baseline模型特征提取失败")
        return
    
    emb_ours = get_embeddings(path_ours, all_texts)
    if emb_ours is None:
        print("❌ Ours模型特征提取失败")
        return

    # 3. t-SNE降维
    print("\n🌌 正在进行t-SNE降维计算...")
    
    # 根据数据量调整perplexity
    perplexity_val = min(30, max(5, len(all_texts) // 5))
    print(f"  使用perplexity = {perplexity_val}")
    
    # 为两个模型分别做t-SNE
    tsne_baseline = TSNE(n_components=2, perplexity=perplexity_val, 
                         random_state=42, init='pca', learning_rate='auto')
    tsne_ours = TSNE(n_components=2, perplexity=perplexity_val,
                     random_state=42, init='pca', learning_rate='auto')
    
    print("  处理baseline模型...")
    pos_baseline_2d = tsne_baseline.fit_transform(emb_baseline)
    
    print("  处理ours模型...")
    pos_ours_2d = tsne_ours.fit_transform(emb_ours)

    # 4. 绘制图表
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 颜色方案 - 为不同aspect分配不同颜色
    unique_aspects = list(set(labels_aspect))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_aspects)))
    aspect_color = {aspect: colors[i] for i, aspect in enumerate(unique_aspects)}
    
    # 极性标记
    markers = {'Positive': 'o', 'Negative': 'X'}
    marker_sizes = {'Positive': 80, 'Negative': 100}  # 负面用稍大的X

    def plot_scatter(ax, data_2d, title, show_legend=False):
        # 为每个aspect和极性绘制点
        for aspect in unique_aspects:
            for polarity in ['Positive', 'Negative']:
                # 找出符合条件的点
                mask = [(a == aspect and p == polarity) 
                       for a, p in zip(labels_aspect, labels_polarity)]
                if any(mask):
                    points = data_2d[mask]
                    ax.scatter(points[:, 0], points[:, 1], 
                              color=aspect_color[aspect],
                              marker=markers[polarity],
                              s=marker_sizes[polarity],
                              alpha=0.7,
                              edgecolors='white',
                              linewidth=0.5,
                              label=f'{ASPECT_MAPPING.get(aspect, aspect)} ({polarity})')
        
        ax.set_title(title, fontweight='bold', size=14, pad=15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle='--', alpha=0.3)
        
        if show_legend:
            # 创建图例
            handles, labels = ax.get_legend_handles_labels()
            # 去重
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), 
                     bbox_to_anchor=(1.05, 1), 
                     loc='upper left',
                     frameon=True,
                     fontsize=9)

    plot_scatter(ax1, pos_baseline_2d, "(a) Baseline模型\n特征分布", show_legend=False)
    plot_scatter(ax2, pos_ours_2d, "(b) Ours模型 (ZRCP + Mask)\n特征分布", show_legend=True)

    plt.suptitle('t-SNE可视化: 不同Aspect的特征分布对比', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图片
    output_png = f'{output_prefix}_comparison.png'
    output_pdf = f'{output_prefix}_comparison.pdf'
    
    plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    
    print(f"\n✅ 图表已保存:")
    print(f"  - {output_png}")
    print(f"  - {output_pdf}")
    
    plt.show()
    
    # 返回统计信息
    return {
        'total_samples': len(all_texts),
        'aspects': unique_aspects,
        'distribution': {ASPECT_MAPPING.get(a, a): sum(1 for x in labels_aspect if x == a) 
                        for a in unique_aspects}
    }

def debug_csv_structure(file_path):
    """调试函数：查看CSV文件的结构"""
    print("\n🔍 调试信息: 检查CSV文件结构")
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    print(f"总行数: {len(df)}")
    print(f"总列数: {len(df.columns)}")
    print("\n前10列:")
    for i, col in enumerate(df.columns[:10]):
        print(f"  {i+1}. {col}")
    
    print("\n前5行数据预览:")
    print(df.head())
    
    # 检查极性值的分布
    print("\n极性值分布检查:")
    for aspect in ['Ambience#Decoration', 'Service#Hospitality', 'Price#Cost_effective']:
        if aspect in df.columns:
            print(f"\n{aspect} 的值分布:")
            print(df[aspect].value_counts().head())

if __name__ == '__main__':
    # 先调试查看CSV结构
    debug_csv_structure("asap-master/data/test.csv")
    
    print("\n" + "="*60)
    print("t-SNE可视化分析")
    print("="*60)
    
    # 可以选择不同的aspect组合进行分析
    aspects_1 = ['Ambience#Decoration', 'Service#Hospitality', 'Food#Taste']
    aspects_2 = ['Ambience#Decoration', 'Ambience#Noise', 'Ambience#Space']
    aspects_3 = ['Price#Level', 'Price#Discount', 'Price#Cost_effective']
    aspects_4 = ['Service#Hospitality', 'Service#Timely', 'Service#Queue']
    
    print("\n请选择要分析的Aspect组合:")
    print("1. 装修装饰 + 服务态度 + 口味")
    print("2. 装修装饰 + 噪音环境 + 空间 (环境相关)")
    print("3. 价格水平 + 折扣优惠 + 性价比 (价格相关)")
    print("4. 服务态度 + 上菜速度 + 排队 (服务相关)")
    print("5. 自定义组合")
    print("6. 调试模式 - 显示CSV结构")
    
    choice = input("\n请输入选择 (1-6): ").strip()
    
    if choice == '6':
        # 已经在上面的debug中显示了
        pass
    elif choice == '1':
        selected = aspects_1
        plot_tsne_comparison(selected_aspects=selected)
    elif choice == '2':
        selected = aspects_2
        plot_tsne_comparison(selected_aspects=selected)
    elif choice == '3':
        selected = aspects_3
        plot_tsne_comparison(selected_aspects=selected)
    elif choice == '4':
        selected = aspects_4
        plot_tsne_comparison(selected_aspects=selected)
    elif choice == '5':
        print("\n可用的Aspect:")
        for i, (key, value) in enumerate(ASPECT_MAPPING.items(), 1):
            print(f"{i}. {key} -> {value}")
        
        indices = input("\n请输入要选择的序号 (用逗号分隔，例如: 1,2,3): ").strip()
        try:
            selected_indices = [int(x.strip()) for x in indices.split(',')]
            selected = [list(ASPECT_MAPPING.keys())[i-1] for i in selected_indices]
            plot_tsne_comparison(selected_aspects=selected)
        except Exception as e:
            print(f"输入无效: {e}")
    else:
        print("使用默认组合")
        plot_tsne_comparison(selected_aspects=aspects_1)