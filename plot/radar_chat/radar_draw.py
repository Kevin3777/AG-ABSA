import numpy as np
import matplotlib.pyplot as plt

def plot_radar_chart():
    # 设置学术风格字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12

    # 选取的 6 个核心 Aspect 标签
    labels = ['Taste', 'Cost-effective', 'Decoration', 
              'Space', 'Parking', 'Recommend']
    num_vars = len(labels)

    # 你的模型 (Ours: ZRCP + Mask) 真实 F1 数据
    ours_scores = [0.9139, 0.9459, 0.8494, 0.8941, 0.8908, 0.9079]
    # 对比基线 (Format C / w/o Mask) 真实 F1 数据
    baseline_scores = [0.8973, 0.9223, 0.8281, 0.8774, 0.8674, 0.8825]

    # 为了画闭合的多边形，需要把第一个值加到最后
    ours_scores += ours_scores[:1]
    baseline_scores += baseline_scores[:1]
    
    # 计算每个轴的角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # 初始化极坐标画布
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    # 绘制 Baseline (虚线 + 浅色填充)
    ax.plot(angles, baseline_scores, color='#E94A4A', linewidth=2, linestyle='--', label='Baseline (w/o Mask)')
    ax.fill(angles, baseline_scores, color='#E94A4A', alpha=0.1)

    # 绘制 Ours (实线 + 深色填充)
    ax.plot(angles, ours_scores, color='#4A90E2', linewidth=2.5, label='Ours (ZRCP + Mask)')
    ax.fill(angles, ours_scores, color='#4A90E2', alpha=0.25)

    # 优化网格和标签
    ax.set_theta_offset(np.pi / 2) # 让第一个标签在正上方
    ax.set_theta_direction(-1)     # 顺时针排列
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontweight='bold')
    
    # 设置雷达图的雷达圈刻度范围 (聚焦在 0.80 到 0.96 之间，让差距更明显)
    ax.set_ylim(0.80, 0.96)
    ax.set_yticks([0.82, 0.86, 0.90, 0.94])
    ax.set_yticklabels(['0.82', '0.86', '0.90', '0.94'], color='gray', size=10)

    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Performance Expansion on High-Frequency Aspects", pad=30, fontweight='bold', size=14)

    plt.tight_layout()
    plt.savefig('radar_chart.pdf', format='pdf', bbox_inches='tight')
    print("✅ 雷达图已保存为: radar_chart.pdf")
    plt.show()

if __name__ == '__main__':
    plot_radar_chart()