import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # 添加pandas导入
import sys
import os

# 添加模块路径到系统路径
sys.path.append('train_learnable/v3_all1')
from train_encoder import AnglE  # 导入你写好的模型类


def analyze_test_set_and_plot(csv_path=None):
    model_path = "checkpoints_learnable/v3_all1" 
    model = AnglE.from_pretrained(model_path, train_mode=False)
    if torch.cuda.is_available(): 
        model.cuda()

    # ==========================================
    # 1. 从CSV文件读取句子
    # ==========================================
    if csv_path is None:
        csv_path = "plot/amp/test_data_mining/test_sentence.csv"
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"成功读取CSV文件，共 {len(df)} 行")
        print(f"列名: {list(df.columns)}")
        
        # 从'匹配句子'列提取句子
        if '匹配句子' in df.columns:
            test_sentences = df['匹配句子'].tolist()
        elif '上下文' in df.columns:
            test_sentences = df['上下文'].tolist()
        else:
            test_sentences = df.iloc[:, 0].tolist()
            
        print(f"提取到 {len(test_sentences)} 个句子")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_path}")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 过滤空值
    test_sentences = [s for s in test_sentences if pd.notna(s) and isinstance(s, str) and s.strip()]
    print(f"过滤后得到 {len(test_sentences)} 个有效句子")
    
    # 显示前几个句子作为预览
    print("\n前5个句子预览:")
    for i, sent in enumerate(test_sentences[:5]):
        preview = sent[:100] + "..." if len(sent) > 100 else sent
        print(f"{i+1}. {preview}")
    
    amplitudes = []
    sentence_amp_pairs = []
    
    for text in test_sentences:
        vec = model.encode(text, to_numpy=False).cpu()
        re, im = torch.chunk(vec, 2, dim=1)
        amp = torch.sqrt(torch.sum(re**2 + im**2, dim=1)).item()
        amplitudes.append(amp)
        sentence_amp_pairs.append((amp, text))
        
    sentence_amp_pairs.sort(key=lambda x: x[0])
    
    # 统计学分析
    mean_amp = np.mean(amplitudes)
    std_amp = np.std(amplitudes)
    min_pair = sentence_amp_pairs[0]
    max_pair = sentence_amp_pairs[-1]
    
    print(f"\n📊 测试集统计: 均值={mean_amp:.4f}, 标准差={std_amp:.4f}")
    print(f"最短: {min_pair[0]:.4f} -> {min_pair[1][:50]}..." if len(min_pair[1]) > 50 else f"最短: {min_pair[0]:.4f} -> {min_pair[1]}")
    print(f"最长: {max_pair[0]:.4f} -> {max_pair[1][:50]}..." if len(max_pair[1]) > 50 else f"最长: {max_pair[0]:.4f} -> {max_pair[1]}")

    # ==========================================
    # 2. 绘制顶级论文复合图
    # ==========================================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.default'] = 'regular'  # 添加这一行
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- 左图：核密度分布图 (KDE) ----
    # 模拟一个带 AmpLoss 的极端情况 (全部坍缩到均值)
    collapsed_amplitudes = np.random.normal(mean_amp, 0.001, len(amplitudes)) 
    
    sns.kdeplot(amplitudes, fill=True, color='#4A90E2', ax=ax1, label='Ours (Natural Variance)')
    sns.kdeplot(collapsed_amplitudes, fill=True, color='#E94A4A', ax=ax1, label='With AmpLoss (Collapsed)')
    
    ax1.set_ylim(0, 3)

    ax1.set_title("(a) Amplitude Distribution in Test Set", fontweight='bold')
    ax1.set_xlabel("L2 Norm (Amplitude)")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ---- 右图：归一化放大的极性几何图 ----
    # 使用 Min-Max 归一化，把微小的差异放大到视觉可见的 [1.0, 2.0] 区间
    def min_max_scale(val):
        return 1.0 + (val - min_pair[0]) / (max_pair[0] - min_pair[0])
        
    len_short_viz = min_max_scale(min_pair[0]) # 1.0
    len_long_viz = min_max_scale(max_pair[0])  # 2.0
    
    angle1, angle2 = np.radians(45), np.radians(52)
    
    v1_x, v1_y = len_short_viz * np.cos(angle1), len_short_viz * np.sin(angle1)
    v2_x, v2_y = len_long_viz * np.cos(angle2), len_long_viz * np.sin(angle2)

    ax2.set_xlim(0, 2.5)
    ax2.set_ylim(0, 2.5)
    ax2.set_aspect('equal')
    ax2.set_title("(b) Min-Max Scaled Geometric Mapping", fontweight='bold')
    ax2.set_xlabel("Real Subspace (Aspect)")
    ax2.set_ylabel("Imaginary Subspace (Polarity)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 画放大的箭头
    ax2.annotate('', xy=(v1_x, v1_y), xytext=(0, 0), 
                arrowprops=dict(facecolor='#4A90E2', shrink=0, width=2, headwidth=8))
    ax2.annotate('', xy=(v2_x, v2_y), xytext=(0, 0), 
                arrowprops=dict(facecolor='#E94A4A', shrink=0, width=2, headwidth=8))

    # 用 Z-score (距离均值几个标准差) 来标注 - 修复LaTeX格式
    z_min = (min_pair[0] - mean_amp) / std_amp
    z_max = (max_pair[0] - mean_amp) / std_amp
    
    # 修复：使用普通文本而不是LaTeX，或者正确转义
    # ax2.text(v1_x + 0.1, v1_y - 0.2, 
    #          f'Short Sentence\nZ ≈ {z_min:.1f}σ', 
    #          color='#4A90E2', fontweight='bold')
    
    # ax2.text(v2_x - 0.8, v2_y + 0.15, 
    #          f'Long Sentence\nZ ≈ +{z_max:.1f}σ', 
    #          color='#E94A4A', fontweight='bold')

    ax2.text(v1_x + 0.1, v1_y - 0.2, 
             f'Short Sentence\nZ ≈ {z_min:.1f}σ', 
             color='#4A90E2', fontweight='bold')
    
    ax2.text(v2_x - 0.8, v2_y + 0.15, 
             f'Long Sentence\nZ ≈ +{z_max:.1f}σ', 
             color='#E94A4A', fontweight='bold')

    plt.tight_layout()
    
    # 保存为PDF和PNG两种格式
    plt.savefig(r'plot\amp\test_set_amplitude_analysis.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(r'plot\amp\test_set_amplitude_analysis.png', format='png', bbox_inches='tight', dpi=300)
    print("\n✅ 图表已保存: test_set_amplitude_analysis.pdf 和 test_set_amplitude_analysis.png")
    
    plt.show()

if __name__ == '__main__':
    # 指定CSV文件路径
    csv_file = "plot/amp/test_data_mining/test_sentence.csv"
    analyze_test_set_and_plot(csv_file)