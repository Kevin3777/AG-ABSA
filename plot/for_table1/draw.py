import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================== 原始数据（保持原始顺序）====================
original_aspects = [
    'Transportation', 'Downtown', 'Easy to find', 'Queue', 'Hospitality',
    'Parking', 'Timely', 'Price Level', 'Cost-effective', 'Discount',
    'Decoration', 'Noise', 'Space', 'Sanitary', 'Portion', 'Taste',
    'Appearance', 'Recommend'   # 原始顺序：Taste -> Appearance -> Recommend
]

f1_data = {
    'RoBERTa': [0.6291, 0.5715, 0.6787, 0.7332, 0.8607, 0.6156, 0.7413, 0.7427,
                0.7439, 0.7012, 0.6815, 0.7272, 0.7431, 0.7705, 0.7220, 0.8486,
                0.7234, 0.6695],
    'CoSENT': [0.7972, 0.6890, 0.8525, 0.7523, 0.8807, 0.7704, 0.8678, 0.8049,
               0.8117, 0.6993, 0.7004, 0.7990, 0.8104, 0.8094, 0.8170, 0.8651,
               0.7426, 0.7814],
    'SimCSE': [0.8517, 0.7663, 0.9189, 0.8206, 0.9393, 0.8798, 0.9172, 0.9193,
               0.9431, 0.7251, 0.8181, 0.9084, 0.8563, 0.8694, 0.8719, 0.8966,
               0.8171, 0.8882],
    'AnglE': [0.8409, 0.8171, 0.9100, 0.8118, 0.9405, 0.8674, 0.9366, 0.9194,
              0.9223, 0.7744, 0.8281, 0.9306, 0.8774, 0.8744, 0.8776, 0.8973,
              0.8366, 0.8825],
    'Ours': [0.8790, 0.7372, 0.9382, 0.8413, 0.9429, 0.8908, 0.9330, 0.9223,
             0.9459, 0.7893, 0.8494, 0.9294, 0.8941, 0.8857, 0.8990, 0.9139,
             0.8327, 0.9079]
}

# ==================== 按原始顺序创建 DataFrame ====================
df_original = pd.DataFrame(f1_data, index=original_aspects)

# ==================== 定义新的方面顺序（最后三个调整为 Recommend, Taste, Appearance）====================
new_aspects = [
    'Transportation', 'Downtown', 'Easy to find', 'Queue', 'Hospitality',
    'Parking', 'Timely', 'Price Level', 'Cost-effective', 'Discount',
    'Decoration', 'Noise', 'Space', 'Sanitary', 'Portion',
    'Recommend',   # 新顺序第一位
    'Taste',       # 新顺序第二位
    'Appearance'   # 新顺序第三位
]

# 重新索引 DataFrame 以匹配新顺序
df = df_original.reindex(new_aspects)

# ==================== 设置基线（RoBERTa） ====================
baseline = df['RoBERTa']
diff_df = df.sub(baseline, axis=0)          # 所有模型减去 RoBERTa 得分

# ==================== 符号映射 ====================
symbol_map = {
    'RoBERTa': '●',
    'CoSENT': '▲',
    'SimCSE': '■',
    'AnglE': '◆',
    'Ours': '★'
}

# ==================== 科研配色 ====================
colors = [
    '#66C2A5',  # RoBERTa (柔和绿)
    '#FC8D62',  # CoSENT   (柔和橙)
    '#8DA0CB',  # SimCSE   (柔和紫)
    '#E78AC3',  # AnglE    (柔和粉)
    '#D55E00'   # Ours     (醒目橙红)
]

# ==================== 绘图 ====================
fig, ax = plt.subplots(figsize=(16, 7))
x = np.arange(len(new_aspects))
width = 0.15

# 绘制柱子，图例标签：RoBERTa 不设 label，其他包含模型名称和对应符号
for i, model in enumerate(df.columns):
    offset = (i - 2) * width          # 使 Ours (索引4) 居中
    label = None if model == 'RoBERTa' else f"{model} {symbol_map[model]}"
    bars = ax.bar(x + offset, diff_df[model], width,
                  label=label,
                  color=colors[i], edgecolor='black', linewidth=0.5)

# 零水平参考线
ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.7)

# ==================== 标记每个方面的最优模型 ====================
best_model_per_aspect = df.idxmax(axis=1)   # 原始得分最高的模型

for idx, aspect in enumerate(new_aspects):
    best_model = best_model_per_aspect[aspect]
    model_index = list(df.columns).index(best_model)
    offset = (model_index - 2) * width
    bar_height = diff_df.loc[aspect, best_model]
    # 根据模型设置不同字号：星号大一点，其他小一点
    if best_model == 'Ours':
        fontsize = 14
    else:
        fontsize = 10
    ax.text(x[idx] + offset, bar_height + 0.001,
            symbol_map[best_model],
            ha='center', va='bottom', fontsize=fontsize,
            color='black', fontweight='bold')

# ==================== 装饰 ====================
ax.set_ylabel('Difference from RoBERTa (F1)', fontsize=14)
ax.set_title('Model Performance Relative to RoBERTa\n(★ marks Ours as best; smaller symbols mark other best models)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(new_aspects, rotation=45, ha='right', fontsize=10)

# 图例放在图内右上角，避免遮挡柱子
ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=11, frameon=True)

ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('diff_from_roberta_symbols_sized.png', dpi=300, bbox_inches='tight')
plt.show()