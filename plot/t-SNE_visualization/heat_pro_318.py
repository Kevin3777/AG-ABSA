import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from matplotlib.gridspec import GridSpec

sys.path.append('train_learnable/v3_all1')
from train_encoder import AnglE

def generate_similarity_heatmap():
    # 模型路径（请根据实际情况修改）
    path_baseline = "checkpoints_standard_angle/v1"
    path_ours = "checkpoints_learnable/v3_all1"

    # 测试文本（10句：5正5负）
    texts = [
        "味道惊艳，回味无穷，绝对是我吃过最好吃的！",
        "完美！口感层次丰富，每一口都是享受，太赞了！",
        "太绝了，从第一口到最后一口都超级满足，人间美味！",
        "无可挑剔，色香味俱全，强烈推荐给所有人！",
        "满分好评，味道太赞了，下次一定还来，超预期！",
        "太难吃了，像嚼蜡一样，完全没味道，恶心！",
        "难以下咽，食材不新鲜，有怪味，吃一口就想吐。",
        "太失望了，味道奇怪，根本吃不下去，浪费钱。",
        "差评！又咸又腻，吃完反胃，绝对不会再来。",
        "口感极差，像是在吃塑料，太难吃了，后悔死了。"
    ]

    def get_similarity_matrix(model_path, texts):
        print(f"  加载模型: {model_path}")
        model = AnglE.from_pretrained(model_path, train_mode=False)
        if torch.cuda.is_available():
            model.cuda()
        embeddings = []
        for text in texts:
            with torch.no_grad():
                vec = model.encode(text, to_numpy=True)[0]
                vec = vec / np.linalg.norm(vec)
                embeddings.append(vec)
        embeddings = np.array(embeddings)
        return np.dot(embeddings, embeddings.T)

    print("="*60)
    print("🧠 提取 Baseline 相似度矩阵...")
    sim_base = get_similarity_matrix(path_baseline, texts)

    print("\n🧠 提取 Ours 相似度矩阵...")
    sim_ours = get_similarity_matrix(path_ours, texts)

    # 计算区域均值
    pos_pos_base = np.mean(sim_base[:5, :5])
    pos_pos_ours = np.mean(sim_ours[:5, :5])
    neg_neg_base = np.mean(sim_base[5:, 5:])
    neg_neg_ours = np.mean(sim_ours[5:, 5:])
    pos_neg_base = np.mean(sim_base[:5, 5:])
    pos_neg_ours = np.mean(sim_ours[:5, 5:])

    diff = sim_ours - sim_base

    # 绘图设置
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['figure.dpi'] = 150

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor('white')

    # 网格：左列两行（Baseline和Ours），右列一整行（Improvement）
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.2], height_ratios=[1, 1],
                  hspace=0.3, wspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])  # Baseline
    ax2 = fig.add_subplot(gs[1, 0])  # Ours
    ax3 = fig.add_subplot(gs[:, 1])  # Improvement

    labels = [f"P{i}" for i in range(1,6)] + [f"N{i}" for i in range(1,6)]

    # ========== Baseline ==========
    sns.heatmap(sim_base, ax=ax1, cmap='RdBu_r', vmin=0.5, vmax=1.0, center=0.75,
                xticklabels=labels, yticklabels=labels, annot=False,
                square=True, cbar=False, linewidths=0.5, linecolor='white')
    ax1.axhline(5, color='white', linewidth=3)
    ax1.axvline(5, color='white', linewidth=3)
    ax1.set_title("(a) Baseline (w/o Mask)", fontweight='bold', pad=15)

    # 区域标注（两行显示）
    ax1.text(0.25, 0.85, f'Pos-Pos:\n{pos_pos_base:.3f}', transform=ax1.transAxes,
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    ax1.text(0.75, 0.85, f'Pos-Neg:\n{pos_neg_base:.3f}', transform=ax1.transAxes,
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    ax1.text(0.25, 0.15, f'Neg-Pos:\n{pos_neg_base:.3f}', transform=ax1.transAxes,
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    ax1.text(0.75, 0.15, f'Neg-Neg:\n{neg_neg_base:.3f}', transform=ax1.transAxes,
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # ========== Ours ==========
    sns.heatmap(sim_ours, ax=ax2, cmap='RdBu_r', vmin=0.5, vmax=1.0, center=0.75,
                xticklabels=labels, yticklabels=labels, annot=False,
                square=True, cbar=False, linewidths=0.5, linecolor='white')
    ax2.axhline(5, color='white', linewidth=3)
    ax2.axvline(5, color='white', linewidth=3)
    ax2.set_title("(b) Ours (ZRCP + Mask)", fontweight='bold', pad=15)

    ax2.text(0.25, 0.85, f'Pos-Pos:\n{pos_pos_ours:.3f}', transform=ax2.transAxes,
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    ax2.text(0.75, 0.85, f'Pos-Neg:\n{pos_neg_ours:.3f}', transform=ax2.transAxes,
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    ax2.text(0.25, 0.15, f'Neg-Pos:\n{pos_neg_ours:.3f}', transform=ax2.transAxes,
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    ax2.text(0.75, 0.15, f'Neg-Neg:\n{neg_neg_ours:.3f}', transform=ax2.transAxes,
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # ========== Improvement ==========
    heatmap = sns.heatmap(diff, ax=ax3, cmap='RdBu', center=0, vmin=-0.2, vmax=0.2,
                          xticklabels=labels, yticklabels=labels, annot=False,
                          square=True, cbar=True, linewidths=0.5, linecolor='white',
                          cbar_kws={"shrink": 0.8, "label": "Δ Similarity (Ours - Baseline)", "pad": 0.02})
    ax3.axhline(5, color='white', linewidth=3)
    ax3.axvline(5, color='white', linewidth=3)
    ax3.set_title("(c) Improvement", fontweight='bold', pad=15)

    # 标注四个区域的差值均值（两行显示）
    reg_names = ['Pos-Pos', 'Pos-Neg', 'Neg-Pos', 'Neg-Neg']
    reg_means = [np.mean(diff[:5,:5]), np.mean(diff[:5,5:]), np.mean(diff[5:,:5]), np.mean(diff[5:,5:])]
    pos_centers = [(0.25,0.75), (0.75,0.75), (0.25,0.25), (0.75,0.25)]
    for name, mean, (x,y) in zip(reg_names, reg_means, pos_centers):
        color = '#2166ac' if mean < 0 else '#b2182b' if mean > 0 else 'gray'
        ax3.text(x, y, f'{name}:\n{mean:+.3f}', transform=ax3.transAxes,
                 ha='center', va='center', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                           edgecolor=color, linewidth=2))

    # ========== 添加共享 colorbar 给左侧两图 ==========
    cbar_ax = fig.add_axes([0.45, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    norm = plt.Normalize(vmin=0.5, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, label='Similarity')
    cbar.ax.yaxis.label.set_size(12)

    # ========== 全局标题和总结 ==========
    fig.suptitle('Similarity Matrix Analysis: Baseline vs Ours', fontsize=18, fontweight='bold', y=0.98)

    impr_pct = (pos_neg_base - pos_neg_ours) / pos_neg_base * 100
    summary = (f"Inter-polarity similarity ↓ {impr_pct:.1f}% ({pos_neg_base:.3f} → {pos_neg_ours:.3f}) | "
               f"Intra-Pos: {pos_pos_ours-pos_pos_base:+.3f} | Intra-Neg: {neg_neg_ours-neg_neg_base:+.3f}")
    fig.text(0.5, 0.02, summary, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8, edgecolor='gray'))

    plt.savefig('similarity_analysis_improved.png', dpi=300, bbox_inches='tight')
    plt.savefig('similarity_analysis_improved.pdf', format='pdf', bbox_inches='tight')
    print("\n✅ 优化版图表（两行标注）已保存")
    plt.show()

if __name__ == '__main__':
    generate_similarity_heatmap()