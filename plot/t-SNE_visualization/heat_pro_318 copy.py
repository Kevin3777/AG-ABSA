import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from matplotlib.gridspec import GridSpec

sys.path.append('train_learnable/v3_all1')
from train_encoder import AnglE  # 导入你写好的模型类

def generate_similarity_heatmap():
    # 替换为你那个 F1=0.8756 (无 Mask / Format C) 的模型路径
    path_baseline = "checkpoints_standard_angle/v1"
    # 替换为你那个 F1=0.8851 (SOTA) 的模型路径
    path_ours = "checkpoints_learnable/v3_all1"

    # ==========================================
    # 1. 精心挑选测试文本 (更极端的正负对立)
    # ==========================================
    texts = [
        # 5 句极端正面 - 口味
        "味道惊艳，回味无穷，绝对是我吃过最好吃的！",
        "完美！口感层次丰富，每一口都是享受，太赞了！",
        "太绝了，从第一口到最后一口都超级满足，人间美味！",
        "无可挑剔，色香味俱全，强烈推荐给所有人！",
        "满分好评，味道太赞了，下次一定还来，超预期！",

        # 5 句极端负面 - 口味
        "太难吃了，像嚼蜡一样，完全没味道，恶心！",
        "难以下咽，食材不新鲜，有怪味，吃一口就想吐。",
        "太失望了，味道奇怪，根本吃不下去，浪费钱。",
        "差评！又咸又腻，吃完反胃，绝对不会再来。",
        "口感极差，像是在吃塑料，太难吃了，后悔死了。"
    ]

    # ==========================================
    # 2. 提取特征的辅助函数
    # ==========================================
    def get_similarity_matrix(model_path, texts):
        print(f"  加载模型: {model_path}")
        model = AnglE.from_pretrained(model_path, train_mode=False)
        if torch.cuda.is_available():
            model.cuda()
            print("  使用GPU加速")

        embeddings = []
        for i, text in enumerate(texts):
            with torch.no_grad():
                vec = model.encode(text, to_numpy=True)[0]
                vec = vec / np.linalg.norm(vec)  # 归一化
                embeddings.append(vec)
            if (i+1) % 5 == 0:
                print(f"  已处理 {i+1}/{len(texts)} 条文本")

        embeddings = np.array(embeddings)
        sim_matrix = np.dot(embeddings, embeddings.T)
        return sim_matrix

    print("=" * 60)
    print("🧠 正在提取 Baseline 的相似度矩阵...")
    print("=" * 60)
    sim_baseline = get_similarity_matrix(path_baseline, texts)

    print("\n" + "=" * 60)
    print("🧠 正在提取 Ours 的相似度矩阵...")
    print("=" * 60)
    sim_ours = get_similarity_matrix(path_ours, texts)

    # ==========================================
    # 3. 计算关键区域的均值
    # ==========================================
    pos_pos_baseline = np.mean(sim_baseline[:5, :5])
    pos_pos_ours = np.mean(sim_ours[:5, :5])
    neg_neg_baseline = np.mean(sim_baseline[5:, 5:])
    neg_neg_ours = np.mean(sim_ours[5:, 5:])
    pos_neg_baseline = np.mean(sim_baseline[:5, 5:])
    pos_neg_ours = np.mean(sim_ours[:5, 5:])

    diff_matrix = sim_ours - sim_baseline

    # ==========================================
    # 4. 绘制三合一拼接图（科研风格）
    # ==========================================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['figure.dpi'] = 150

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('white')

    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.2],
                  height_ratios=[1, 1], hspace=0.25, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])  # Baseline
    ax2 = fig.add_subplot(gs[1, 0])  # Ours
    ax3 = fig.add_subplot(gs[:, 1])  # Improvement

    labels = [f"P{i}" for i in range(1, 6)] + [f"N{i}" for i in range(1, 6)]

    # ===== 左上：Baseline =====
    sns.heatmap(sim_baseline, ax=ax1, cmap='RdBu_r',
                vmin=0.5, vmax=1.0, center=0.75,
                xticklabels=labels, yticklabels=labels,
                annot=False,  # 不显示具体数值
                square=True, cbar=False,
                linewidths=0.5, linecolor='white')
    ax1.axhline(5, color='white', linewidth=3)
    ax1.axvline(5, color='white', linewidth=3)
    ax1.set_title("(a) Baseline (w/o Mask)", fontweight='bold', pad=15)

    # 区域均值标注
    ax1.text(0.25, 0.85, f'Pos-Pos: {pos_pos_baseline:.3f}',
             transform=ax1.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    ax1.text(0.75, 0.85, f'Pos-Neg: {pos_neg_baseline:.3f}',
             transform=ax1.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    ax1.text(0.25, 0.15, f'Neg-Pos: {pos_neg_baseline:.3f}',
             transform=ax1.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    ax1.text(0.75, 0.15, f'Neg-Neg: {neg_neg_baseline:.3f}',
             transform=ax1.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # ===== 左下：Ours =====
    sns.heatmap(sim_ours, ax=ax2, cmap='RdBu_r',
                vmin=0.5, vmax=1.0, center=0.75,
                xticklabels=labels, yticklabels=labels,
                annot=False,
                square=True, cbar=False,
                linewidths=0.5, linecolor='white')
    ax2.axhline(5, color='white', linewidth=3)
    ax2.axvline(5, color='white', linewidth=3)
    ax2.set_title("(b) Ours (ZRCP + Mask)", fontweight='bold', pad=15)

    ax2.text(0.25, 0.85, f'Pos-Pos: {pos_pos_ours:.3f}',
             transform=ax2.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    ax2.text(0.75, 0.85, f'Pos-Neg: {pos_neg_ours:.3f}',
             transform=ax2.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    ax2.text(0.25, 0.15, f'Neg-Pos: {pos_neg_ours:.3f}',
             transform=ax2.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    ax2.text(0.75, 0.15, f'Neg-Neg: {neg_neg_ours:.3f}',
             transform=ax2.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # ===== 右侧：Improvement =====
    # 使用 'RdBu' 红色高、蓝色低，使负值（改进）显示为蓝色
    sns.heatmap(diff_matrix, ax=ax3, cmap='RdBu',
                center=0, vmin=-0.2, vmax=0.2,
                xticklabels=labels, yticklabels=labels,
                annot=False,
                square=True,
                cbar_kws={"shrink": 0.8, "label": "Δ Similarity (Ours - Baseline)", "pad": 0.02},
                linewidths=0.5, linecolor='white')
    ax3.axhline(5, color='white', linewidth=3)
    ax3.axvline(5, color='white', linewidth=3)
    ax3.set_title("(c) Improvement (Ours - Baseline)", fontweight='bold', pad=15)

    # 在Improvement图上标注区域均值
    region_names = ['Pos-Pos', 'Pos-Neg', 'Neg-Pos', 'Neg-Neg']
    region_means = [
        np.mean(diff_matrix[:5, :5]),
        np.mean(diff_matrix[:5, 5:]),
        np.mean(diff_matrix[5:, :5]),
        np.mean(diff_matrix[5:, 5:])
    ]
    positions = [(0.25, 0.75), (0.75, 0.75), (0.25, 0.25), (0.75, 0.25)]
    for name, mean, pos in zip(region_names, region_means, positions):
        color = '#4575b4' if mean < 0 else '#d73027' if mean > 0 else '#ffffbf'
        ax3.text(pos[0], pos[1], f'{name}\n{mean:+.3f}',
                 transform=ax3.transAxes, ha='center', va='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                           edgecolor=color, linewidth=2))

    # ===== 全局标题和关键发现 =====
    fig.suptitle('Similarity Matrix Analysis: Baseline vs Ours', fontsize=16, fontweight='bold', y=0.98)

    # 底部总结
    improvement = (pos_neg_baseline - pos_neg_ours) / pos_neg_baseline * 100
    summary = (f"Inter-polarity similarity reduced by {improvement:.1f}% "
               f"({pos_neg_baseline:.3f} → {pos_neg_ours:.3f}), "
               f"intra-class cohesion slightly decreased (Pos-Pos: {pos_pos_ours-pos_pos_baseline:+.3f}, "
               f"Neg-Neg: {neg_neg_ours-neg_neg_baseline:+.3f})")
    fig.text(0.5, 0.02, summary, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8, edgecolor='gray'))

    plt.savefig('similarity_analysis_final.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('similarity_analysis_final.pdf', format='pdf', bbox_inches='tight', facecolor='white')
    print("\n✅ 科研风格热力图已保存: similarity_analysis_final.png/pdf")
    plt.show()

if __name__ == '__main__':
    generate_similarity_heatmap()