import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import sys
import os

# 添加模块路径到系统路径
sys.path.append('train_learnable/v3_all1')
from train_encoder import AnglE  # 导入你写好的模型类

def generate_tsne_plot():
    # ==========================================
    # 1. 准备你要提取的两个模型路径
    # ==========================================
    # 替换为你那个 F1=0.8756 (无 Mask / Format C) 的模型路径
    path_baseline = "checkpoints_standard_angle/v1" 
    # 替换为你那个 F1=0.8851 (SOTA) 的模型路径
    path_ours = "checkpoints_learnable/v3_all1"

    # ==========================================
    # 2. 准备测试文本 (精选 3 个容易混淆的 Aspect，分别带正负极性)
    # ==========================================
    # 为了让 t-SNE 点足够多，每个类别我建议你自己多复制/补充几句话
    texts_taste_pos = ["口味绝了，非常好吃！", "味道很赞，强烈推荐。", "菜品色香味俱全，口感一流。"] * 10 
    texts_taste_neg = ["太难吃了，简直难以下咽。", "味道很奇怪，不符合口味。", "菜都馊了，一股怪味。"] * 10
    
    texts_service_pos = ["服务员态度特别好，很贴心。", "服务非常周到，随叫随到。", "前台小姐姐很热情。"] * 10
    texts_service_neg = ["服务态度极差，爱答不理的。", "等了半天上菜，催了也没人管。", "服务员全程黑脸。"] * 10
    
    texts_price_pos = ["性价比超高，非常划算。", "价格很亲民，分量又大。", "打折下来很便宜。"] * 10
    texts_price_neg = ["太贵了，简直是抢钱。", "分量极小，完全不值这个价。", "性价比极低，坑人。"] * 10

    all_texts = texts_taste_pos + texts_taste_neg + texts_service_pos + texts_service_neg + texts_price_pos + texts_price_neg
    
    # 制作对应的颜色/形状标签
    labels_aspect = ['Taste'] * 60 + ['Service'] * 60 + ['Price'] * 60
    labels_polarity = (['Positive']*30 + ['Negative']*30) * 3

    # ==========================================
    # 3. 提取特征的辅助函数
    # ==========================================
    def get_embeddings(model_path, texts):
        print(f"正在加载模型: {model_path} ...")
        model = AnglE.from_pretrained(model_path, train_mode=False)
        if torch.cuda.is_available(): model.cuda()
        
        embeddings = []
        # 逐句提取特征
        for text in texts:
            with torch.no_grad():
                # 提取通过 ZRCP 或者直接输出的特征
                vec = model.encode(text, to_numpy=True)
                embeddings.append(vec[0]) 
        return np.array(embeddings)

    # 提取特征
    emb_baseline = get_embeddings(path_baseline, all_texts)
    emb_ours = get_embeddings(path_ours, all_texts)

    # ==========================================
    # 4. t-SNE 降维
    # ==========================================
    print("正在进行 t-SNE 降维计算 (可能需要几秒钟)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    
    pos_baseline_2d = tsne.fit_transform(emb_baseline)
    pos_ours_2d = tsne.fit_transform(emb_ours)

    # ==========================================
    # 5. 绘制并排双图
    # ==========================================
    plt.rcParams['font.family'] = 'serif'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 颜色映射 (Aspect: 红色系/蓝色系/绿色系)
    palette = {'Taste': '#E94A4A', 'Service': '#4A90E2', 'Price': '#50B848'}
    markers = {'Positive': 'o', 'Negative': 'X'} # 圆点代表正面，叉叉代表负面

    def plot_scatter(ax, data_2d, title):
        for i in range(len(data_2d)):
            aspect = labels_aspect[i]
            polarity = labels_polarity[i]
            ax.scatter(data_2d[i, 0], data_2d[i, 1], 
                       c=palette[aspect], marker=markers[polarity], 
                       s=80, alpha=0.7, edgecolors='white', linewidth=0.5)
        ax.set_title(title, fontweight='bold', size=14, pad=15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    # 绘制左图
    plot_scatter(ax1, pos_baseline_2d, "(a) Baseline (w/o Mask)\nCollapsed Clusters & Entangled Polarities")
    
    # 绘制右图
    plot_scatter(ax2, pos_ours_2d, "(b) Ours (ZRCP + Mask)\nDistinct Aspect Clusters & Polarized Sub-clusters")

    # 手动添加图例 (放在中间下方)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E94A4A', markersize=10, label='Taste (Pos)'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#E94A4A', markersize=10, label='Taste (Neg)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4A90E2', markersize=10, label='Service (Pos)'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#4A90E2', markersize=10, label='Service (Neg)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#50B848', markersize=10, label='Price (Pos)'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#50B848', markersize=10, label='Price (Neg)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.05), frameon=False)

    plt.tight_layout()
    plt.savefig('tsne_visualization.pdf', format='pdf', bbox_inches='tight')
    print("✅ t-SNE 图已保存为: tsne_visualization.pdf")
    plt.show()

if __name__ == '__main__':
    generate_tsne_plot()