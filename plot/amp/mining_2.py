import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import sys
import os

# 将你的模型路径加入系统路径 (与 mining.py 保持一致)
sys.path.append('train_learnable/v3_all1')
from train_encoder import AnglE 

def generate_publication_scatter_plot(csv_path=None):
    # ==========================================
    # 1. 初始化模型
    # ==========================================
    print("⏳ 正在加载模型...")
    model_path = "checkpoints_learnable/v3_all1" 
    model = AnglE.from_pretrained(model_path, train_mode=False)
    if torch.cuda.is_available(): 
        model.cuda()
    print("✅ 模型加载完成！")

    # ==========================================
    # 2. 读取文本数据
    # ==========================================
    if csv_path is None:
        # 默认尝试读取你通过 data_processing.py 生成的带上下文的文件
        csv_path = "test_包含装修_上下文.csv"
        if not os.path.exists(csv_path):
            # 退回 mining.py 中使用的默认路径
            csv_path = "plot/amp/test_data_mining/test_sentence.csv"

    try:
        df_raw = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"📄 成功读取数据文件: {csv_path}，共 {len(df_raw)} 行")
        
        if '上下文' in df_raw.columns:
            test_sentences = df_raw['上下文'].tolist()
        elif '匹配句子' in df_raw.columns:
            test_sentences = df_raw['匹配句子'].tolist()
        else:
            test_sentences = df_raw.iloc[:, 0].tolist()
    except Exception as e:
        print(f"❌ 读取文件出错: {e}")
        return

    # 过滤空值
    test_sentences = [s for s in test_sentences if pd.notna(s) and isinstance(s, str) and s.strip()]
    
    # ==========================================
    # 3. 提取特征：文本长度 & 几何模长
    # ==========================================
    print("⏳ 正在通过模型计算特征，请稍候...")
    lengths = []
    amplitudes = []
    
    for text in test_sentences:
        lengths.append(len(text)) # 物理长度（中文字符数）
        
        # 计算模长 (与 mining.py 逻辑一致)
        vec = model.encode(text, to_numpy=False).cpu()
        re, im = torch.chunk(vec, 2, dim=1)
        amp = torch.sqrt(torch.sum(re**2 + im**2, dim=1)).item()
        amplitudes.append(amp)
        
    df = pd.DataFrame({
        'Text_Length': lengths, 
        'Amplitude': amplitudes,
        'Sentence': test_sentences
    })

    # ==========================================
    # 4. 自动定位关键样本 (用于图表高亮和论文素材)
    # ==========================================
    mean_amp = df['Amplitude'].mean()
    mean_len = df['Text_Length'].mean()

    # 极值样本
    idx_min_amp = df['Amplitude'].idxmin()
    idx_max_amp = df['Amplitude'].idxmax()

    # 均值附近的样本（寻找模长最接近均值的点，挑一个相对长的，一个相对短的）
    df['Dist_to_Mean_Amp'] = abs(df['Amplitude'] - mean_amp)
    df_near_mean = df.sort_values('Dist_to_Mean_Amp').head(20) # 取最接近均值的20个样本
    idx_mean_short = df_near_mean['Text_Length'].idxmin()
    idx_mean_long = df_near_mean['Text_Length'].idxmax()

    highlight_indices = [idx_min_amp, idx_max_amp, idx_mean_long, idx_mean_short]
    highlight_labels = [
        'Min Amp (Objective)', 
        'Max Amp (Subjective)',
        'Avg Amp (Long Context)',
        'Avg Amp (Short Context)'
    ]
    # colors = ['#8172B3', '#C44E52', '#55A868', '#55A868']
    colors = ['#8172B3', '#E69F00', '#55A868', '#8B0000'] 

    print("\n🔍 被自动提取的高亮代表性样本（请将这些加入你的论文 Section 5.6 中）：")
    for idx, label in zip(highlight_indices, highlight_labels):
        print(f"[{label}] 模长={df.loc[idx, 'Amplitude']:.3f}, 长度={df.loc[idx, 'Text_Length']}")
        print(f"文本: {df.loc[idx, 'Sentence']}\n")

    # ==========================================
    # 5. 统计学检验
    # ==========================================
    r, p_value = pearsonr(df['Text_Length'], df['Amplitude'])
    
    # ==========================================
    # 6. 绘制高水平学术图表
    # ==========================================
    print("📊 正在生成图表...")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'legend.fontsize': 11,
        'figure.dpi': 300
    })

    fig, ax = plt.subplots(figsize=(9, 6))

    # 绘制回归散点图
    sns.regplot(
        data=df, x='Text_Length', y='Amplitude', ax=ax,
        scatter_kws={'alpha': 0.4, 's': 35, 'color': '#4C72B0', 'edgecolor': 'w'},
        line_kws={'color': '#C44E52', 'linewidth': 2.5, 'linestyle': '--'}
    )

    # 均值基准线
    ax.axhline(mean_amp, color='gray', linestyle=':', linewidth=2, alpha=0.8, label=f'Mean Amplitude ({mean_amp:.2f})')

    # 标注特殊样本
    # for idx, label, color in zip(highlight_indices, highlight_labels, colors):
    #     x_val = df.loc[idx, 'Text_Length']
    #     y_val = df.loc[idx, 'Amplitude']

    #     ax.scatter(x_val, y_val, color=color, s=250, marker='*', edgecolor='black', zorder=5)
        
    #     offset_y = (df['Amplitude'].max() - df['Amplitude'].min()) * 0.05
    #     offset_y = offset_y if y_val > mean_amp else -offset_y
        
    #     ax.annotate(
    #         label, (x_val, y_val),
    #         xytext=(x_val + (df['Text_Length'].max()*0.02), y_val + offset_y),
    #         fontsize=10, fontweight='bold', color=color,
    #         arrowprops=dict(arrowstyle="->", color='gray', lw=1),
    #         zorder=6
    #     )

    # 标注特殊样本，强制文本在轴框内
    # for i, (idx, label, color) in enumerate(zip(highlight_indices, highlight_labels, colors)):
    #     x_val = df.loc[idx, 'Text_Length']
    #     y_val = df.loc[idx, 'Amplitude']
        
    #     # 获取当前坐标轴范围
    #     x_min, x_max = ax.get_xlim()
    #     y_min, y_max = ax.get_ylim()
    #     x_range = x_max - x_min
    #     y_range = y_max - y_min
        
    #     # 计算偏移量（相对范围的比例）
    #     x_offset = x_range * 0.03
    #     y_offset = y_range * 0.05
        
    #     # 初始假设文本放在点的右上方
    #     x_text = x_val + x_offset
    #     y_text = y_val + y_offset
    #     ha = 'left'
    #     va = 'bottom'
        
    #     # 检查并调整x方向
    #     if x_text > x_max:
    #         x_text = x_val - x_offset
    #         ha = 'right'
    #         # 如果左边也超出，则放在边界内侧
    #         if x_text < x_min:
    #             x_text = x_min + x_range * 0.01
    #             ha = 'left'
        
    #     # 检查并调整y方向
    #     if y_text > y_max:
    #         y_text = y_val - y_offset
    #         va = 'top'
    #         # 如果下边也超出，则放在边界内侧
    #         if y_text < y_min:
    #             y_text = y_min + y_range * 0.01
    #             va = 'bottom'
        
    #     # 如果点在边界附近，进一步微调
    #     if x_val < x_min + x_range * 0.1:  # 点在左边界附近
    #         x_text = x_val + x_offset
    #         ha = 'left'
    #     if x_val > x_max - x_range * 0.1:  # 点在右边界附近
    #         x_text = x_val - x_offset
    #         ha = 'right'
    #     if y_val < y_min + y_range * 0.1:  # 点在下边界附近
    #         y_text = y_val + y_offset
    #         va = 'bottom'
    #     if y_val > y_max - y_range * 0.1:  # 点在上边界附近
    #         y_text = y_val - y_offset
    #         va = 'top'
        
    #     # 最终边界保护
    #     x_text = max(x_min + x_range * 0.01, min(x_max - x_range * 0.01, x_text))
    #     y_text = max(y_min + y_range * 0.01, min(y_max - y_range * 0.01, y_text))
        
    #     if i == 2:  # 如果是绿色点（第四个点）
    #         print(f"⚠️ 绿色点文本已从 ({x_text_original:.1f}, {y_text_original:.1f}) 调整至 ({x_text:.1f}, {y_text:.1f})")
    #     else:
    #         # 不需要调整，使用原位置
    #         x_text = x_text_original
    #         y_text = y_text_original
    #         ha = 'left'
    #         va = 'center'
        
    #     # 绘制高亮标记点
    #     ax.scatter(x_val, y_val, color=color, s=250, marker='*', edgecolor='black', zorder=5)
        
    #     # 添加标注文本，确保在轴内
    #     ax.annotate(
    #         label, (x_val, y_val),
    #         xytext=(x_text, y_text),
    #         textcoords='data',
    #         ha=ha, va=va,
    #         fontsize=10, fontweight='bold', color=color,
    #         arrowprops=dict(arrowstyle="->", color='gray', lw=1),
    #         zorder=6
    #     )


    # for i, (idx, label, color) in enumerate(zip(highlight_indices, highlight_labels, colors)):
    #         x_val = df.loc[idx, 'Text_Length']
    #         y_val = df.loc[idx, 'Amplitude']

    #         # 获取当前坐标轴范围
    #         x_min, x_max = ax.get_xlim()
    #         y_min, y_max = ax.get_ylim()
    #         x_range = x_max - x_min
    #         y_range = y_max - y_min

    #         # 计算偏移量（相对范围的比例）
    #         x_offset = x_range * 0.03
    #         y_offset = y_range * 0.05

    #         # 初始假设文本放在点的右上方
    #         x_text = x_val + x_offset
    #         y_text = y_val + y_offset
    #         ha = 'left'
    #         va = 'bottom'

    #         # 检查并调整 x 方向
    #         if x_text > x_max:
    #             x_text = x_val - x_offset
    #             ha = 'right'
    #             # 如果左边也超出，则放在边界内侧
    #             if x_text < x_min:
    #                 x_text = x_min + x_range * 0.01
    #                 ha = 'left'

    #         # 检查并调整 y 方向
    #         if y_text > y_max:
    #             y_text = y_val - y_offset
    #             va = 'top'
    #             # 如果下边也超出，则放在边界内侧
    #             if y_text < y_min:
    #                 y_text = y_min + y_range * 0.01
    #                 va = 'bottom'

    #         # 如果点在边界附近，进一步微调
    #         if x_val < x_min + x_range * 0.1:  # 点在左边界附近
    #             x_text = x_val + x_offset
    #             ha = 'left'
    #         if x_val > x_max - x_range * 0.1:  # 点在右边界附近
    #             x_text = x_val - x_offset
    #             ha = 'right'
    #         if y_val < y_min + y_range * 0.1:  # 点在下边界附近
    #             y_text = y_val + y_offset
    #             va = 'bottom'
    #         if y_val > y_max - y_range * 0.1:  # 点在上边界附近
    #             y_text = y_val - y_offset
    #             va = 'top'

    #         # 最终边界保护
    #         x_text = max(x_min + x_range * 0.01, min(x_max - x_range * 0.01, x_text))
    #         y_text = max(y_min + y_range * 0.01, min(y_max - y_range * 0.01, y_text))


    #         # 绘制高亮标记点
    #         ax.scatter(x_val, y_val, color=color, s=250, marker='*', edgecolor='black', zorder=5)

    #         # 添加标注文本，确保在轴内
    #         ax.annotate(
    #             label, (x_val, y_val),
    #             xytext=(x_text, y_text),
    #             textcoords='data',
    #             ha=ha, va=va,
    #             fontsize=10, fontweight='bold', color=color,
    #             arrowprops=dict(arrowstyle="->", color='gray', lw=1),
    #             zorder=6
    #         )


    for i, (idx, label, color) in enumerate(zip(highlight_indices, highlight_labels, colors)):
        x_val = df.loc[idx, 'Text_Length']
        y_val = df.loc[idx, 'Amplitude']
        
        # 获取当前坐标轴范围
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # 计算偏移量（相对范围的比例）
        x_offset = x_range * 0.03
        y_offset = y_range * 0.05
        
        # 为绿色点（索引2）单独设置文本位置规则
        if i == 2:  # Avg Amp (Short Context) - 绿色点
            # 将文本放在点的下方
            x_text = x_val
            y_text = y_val - y_offset * 1.5  # 向下偏移
            ha = 'center'
            va = 'top'
            
            # 检查是否超出下边界
            if y_text < y_min:
                y_text = y_min + y_range * 0.01
                va = 'bottom'
            
            # 检查是否超出左右边界
            if x_text < x_min:
                x_text = x_min + x_range * 0.01
                ha = 'left'
            elif x_text > x_max:
                x_text = x_max - x_range * 0.01
                ha = 'right'
        
        else:
            # 其他点使用原来的规则（右上方）
            # 初始假设文本放在点的右上方
            x_text = x_val + x_offset
            y_text = y_val + y_offset
            ha = 'left'
            va = 'bottom'
            
            # 检查并调整 x 方向
            if x_text > x_max:
                x_text = x_val - x_offset
                ha = 'right'
                # 如果左边也超出，则放在边界内侧
                if x_text < x_min:
                    x_text = x_min + x_range * 0.01
                    ha = 'left'
            
            # 检查并调整 y 方向
            if y_text > y_max:
                y_text = y_val - y_offset
                va = 'top'
                # 如果下边也超出，则放在边界内侧
                if y_text < y_min:
                    y_text = y_min + y_range * 0.01
                    va = 'bottom'
            
            # 如果点在边界附近，进一步微调
            if x_val < x_min + x_range * 0.1:  # 点在左边界附近
                x_text = x_val + x_offset
                ha = 'left'
            if x_val > x_max - x_range * 0.1:  # 点在右边界附近
                x_text = x_val - x_offset
                ha = 'right'
            if y_val < y_min + y_range * 0.1:  # 点在下边界附近
                y_text = y_val + y_offset
                va = 'bottom'
            if y_val > y_max - y_range * 0.1:  # 点在上边界附近
                y_text = y_val - y_offset
                va = 'top'
            
            # 最终边界保护
            x_text = max(x_min + x_range * 0.01, min(x_max - x_range * 0.01, x_text))
            y_text = max(y_min + y_range * 0.01, min(y_max - y_range * 0.01, y_text))
        
        # 绘制高亮标记点
        ax.scatter(x_val, y_val, color=color, s=250, marker='*', edgecolor='black', zorder=5)
        
        # 添加标注文本，确保在轴内
        ax.annotate(
            label, (x_val, y_val),
            xytext=(x_text, y_text),
            textcoords='data',
            ha=ha, va=va,
            fontsize=10, fontweight='bold', color=color,
            arrowprops=dict(arrowstyle="->", color='gray', lw=1),
            zorder=6
        )


    # 统计信息文本框
    p_text = f"p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f} (N.S.)"
    stats_text = f"Pearson $r$: {r:.3f}\n{p_text}\n$N = {len(df)}$"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.set_title("Independence of Vector Magnitude and Text Length\n(Verification of Geometric Decoupling)", pad=15, fontweight='bold')
    ax.set_xlabel("Physical Text Length (Number of Characters)")
    ax.set_ylabel(r"$L_2$ Norm (Complex Amplitude)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower right')

    plt.tight_layout()

    # 确保输出目录存在
    os.makedirs('plot/amp', exist_ok=True)
    save_path_pdf = 'plot/amp/Magnitude_vs_TextLength_Verification.pdf'
    save_path_png = 'plot/amp/Magnitude_vs_TextLength_Verification.png'
    
    plt.savefig(save_path_pdf, bbox_inches='tight')
    plt.savefig(save_path_png, bbox_inches='tight', dpi=300)
    
    print(f"✅ 图表已保存至:\n  - {save_path_pdf}\n  - {save_path_png}")
    plt.show()

if __name__ == '__main__':
    # 运行函数，如果你想指定特定的文件，可以在这里修改路径
    # 例如: generate_publication_scatter_plot("asap-master/data/test.csv")
    generate_publication_scatter_plot()