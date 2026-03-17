import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import sys
from sklearn.manifold import TSNE
import plotly.express as px
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# --- 🛠️ 路径自动加固和 AnglE 导入 ---
# 假设脚本在 D:\WorkSpace\AnglE_yj\
PROJECT_ROOT = r'D:\WorkSpace\AnglE_yj' 
ANGLE_ROOT = os.path.join(PROJECT_ROOT, 'AnglE')
if ANGLE_ROOT not in sys.path:
    sys.path.insert(0, ANGLE_ROOT)

try:
    from angle_emb import AnglE
except ImportError:
    st.error("❌ 无法导入 AnglE。请检查路径设置和库安装。")
    st.stop()

# --- 配置参数 (请根据您的实际路径调整) ---
TEST_CSV_FILE = os.path.join(PROJECT_ROOT, r'asap-master\data\test.csv')
VISUALIZATION_OUTPUT_DIR = 'visualization_results'

# ⚠️ 模型路径配置
MODEL_PATH_ANGLE_ALCE = os.path.join(PROJECT_ROOT, r'checkpoints\angle_alce_encoder_final_V2\checkpoint-655')
MODEL_PATH_DOT_PRODUCT = os.path.join(PROJECT_ROOT, r'checkpoint_dotproduct\dot_product_model\best_model')
BASE_MODEL_NAME = "hfl/chinese-roberta-wwm-ext" 

# 所有 Aspect 映射
ASPECT_MAPPING = {
    'Location#Transportation': '交通', 'Location#Downtown': '地段', 'Location#Easy_to_find': '位置易找度',
    'Service#Queue': '排队', 'Service#Hospitality': '服务态度', 'Service#Parking': '停车', 'Service#Timely': '上菜速度',
    'Price#Level': '价格水平', 'Price#Cost_effective': '性价比', 'Price#Discount': '折扣优惠',
    'Ambience#Decoration': '装修装饰', 'Ambience#Noise': '噪音环境', 'Ambience#Space': '空间', 'Ambience#Sanitary': '卫生',
    'Food#Portion': '份量', 'Food#Taste': '口味', 'Food#Appearance': '外观卖相', 'Food#Recommend': '推荐度'
}
ALL_ASPECTS = list(ASPECT_MAPPING.keys())

# --- 核心工具函数 ---

@st.cache_data
def load_data():
    """缓存加载和清洗数据"""
    try:
        df = pd.read_csv(TEST_CSV_FILE)
        df['review'] = df['review'].apply(lambda x: str(x).replace('\n', ' ').strip())
        return df
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_models():
    """缓存加载三个模型"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_angle(path):
        return AnglE.from_pretrained(path, pooling_strategy='cls').to(device)

    def load_bert(path):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path).to(device)
        return tokenizer, model

    st.info("🚀 正在加载 AnglE-ALCE 模型...")
    angle_alce = load_angle(MODEL_PATH_ANGLE_ALCE)
    
    st.info("🚀 正在加载 Dot-Product 模型...")
    dot_product = load_angle(MODEL_PATH_DOT_PRODUCT)

    st.info("🚀 正在加载 Baseline BERT 模型...")
    bert_tokenizer, bert_model = load_bert(BASE_MODEL_NAME)
    
    st.success("✅ 所有模型加载完成并缓存。")
    
    return angle_alce, dot_product, bert_tokenizer, bert_model

def get_embeddings(model_obj, df_subset, model_type="angle", batch_size=64):
    """提取嵌入向量"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = []
    
    for i in tqdm(range(0, len(df_subset), batch_size), desc=f"提取 {model_type.upper()} 嵌入", leave=False):
        batch = df_subset.iloc[i:i + batch_size]
        
        if model_type == "angle":
            texts = [f"Aspect: {ASPECT_MAPPING[row['label_aspect']]}, Context: {row['review']}" 
                     for _, row in batch.iterrows()]
            embeddings.append(model_obj.encode(texts, to_numpy=True))
        else: # bert
            tokenizer, model = model_obj
            texts = batch['review'].tolist()
            inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings.append(outputs[0][:, 0, :].cpu().numpy())

    return np.concatenate(embeddings, axis=0)

def apply_tsne_and_plot(df, vectors, model_name):
    """应用 t-SNE 降维并使用 Plotly 绘图"""
    
    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(len(df) - 1, 30), max_iter=1000)
    X_reduced = tsne.fit_transform(vectors)
    
    # 结果加入 DataFrame
    df['TSNE-1'] = X_reduced[:, 0]
    df['TSNE-2'] = X_reduced[:, 1]
    
    # 绘制交互式散点图
    fig = px.scatter(
        df, 
        x='TSNE-1', 
        y='TSNE-2', 
        color='label_polarity', # 颜色：情感极性
        symbol='label_aspect',   # 形状：Aspect 类型
        color_discrete_map={'正面 (1)': 'blue', '负面 (-1)': 'red'},
        title=f'{model_name}',
        hover_data=['review_short', 'label_aspect', 'label_polarity'],
        height=500
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    fig.update_traces(marker=dict(size=8))
    
    return fig

# --- STREAMLIT APP LAYOUT ---

def main_app():
    st.set_page_config(layout="wide", page_title="AnglE Loss 向量空间可视化")

    # --- 标题 ---
    st.title("🏆 AnglE Loss 可视化演示：特征质量 SOTA 验证")
    st.markdown("该工具用于对比 **AnglE-ALCE**、**点积模型**和 **Baseline BERT** 在细粒度情感分析任务中词向量的 **几何结构 (角度分离)**。")
    st.markdown("---")

    # --- 1. 侧边栏：数据选择 ---
    st.sidebar.header("📊 数据选择与控制")

    # Aspect 选择
    selected_aspects = st.sidebar.multiselect(
        "选择用于可视化的方面 (Aspects):",
        options=ALL_ASPECTS,
        default=['Food#Taste', 'Service#Hospitality'],
        format_func=lambda x: f"{x} ({ASPECT_MAPPING[x]})"
    )

    # 样本数选择
    samples_per_aspect = st.sidebar.slider(
        "每个方面采样的评论条数 (正负各半):",
        min_value=5, max_value=30, value=15
    )
    
    # --- 2. 模型加载 ---
    angle_alce, dot_product, bert_tokenizer, bert_model = load_models()
    
    if not selected_aspects:
        st.warning("请至少选择一个 Aspect 进行可视化。")
        return

    # --- 3. 筛选数据 ---
    df_data = load_data()
    
    filtered_samples = []
    total_samples = 0
    
    # ... (数据筛选逻辑不变) ...
    for aspect_en in selected_aspects:
        aspect_df = df_data[df_data[aspect_en].isin([1, -1])].copy()
        
        positive_samples = aspect_df[aspect_df[aspect_en] == 1].sample(
            min(samples_per_aspect, len(aspect_df[aspect_df[aspect_en] == 1])), 
            random_state=42
        )
        negative_samples = aspect_df[aspect_df[aspect_en] == -1].sample(
            min(samples_per_aspect, len(aspect_df[aspect_df[aspect_en] == -1])),
            random_state=42
        )
        
        subset = pd.concat([positive_samples, negative_samples]).copy()
        subset['label_polarity'] = subset[aspect_en].apply(lambda x: '正面 (1)' if x == 1 else '负面 (-1)')
        subset['label_aspect'] = aspect_en
        subset['review_short'] = subset['review'].apply(lambda x: f"{x[:80]}..." if len(x) > 80 else x)
        
        filtered_samples.append(subset)
        total_samples += len(subset)

    final_df = pd.concat(filtered_samples).reset_index(drop=True)
    
    st.sidebar.info(f"总计用于可视化的样本数: **{total_samples}**")
    
    if total_samples < 5:
        st.warning("样本数太少，t-SNE 无法有效运行，请增加样本数。")
        return

    # --- 4. 提取向量 (使用 Streamlit 进度条) ---
    st.subheader("特征向量提取与降维")
    with st.spinner('正在提取和降维词向量...'):
        
        # 提取 AnglE-ALCE 向量
        vectors_angle_alce = get_embeddings(angle_alce, final_df, model_type="angle")
        df_angle_alce = final_df.copy()

        # 提取 Dot-Product 向量
        vectors_dot_product = get_embeddings(dot_product, final_df, model_type="angle")
        df_dot_product = final_df.copy()

        # 提取 Baseline BERT 向量
        vectors_bert = get_embeddings((bert_tokenizer, bert_model), final_df, model_type="bert")
        df_bert = final_df.copy()

    st.success("✅ 向量提取和降维完成。")

    # --- 5. 主面板：可视化对比 ---
    st.header("词向量空间对比 (t-SNE)")
    st.markdown("💡 **AnglE-ALCE** 应显示出**更紧密的聚类**和**更清晰的 Aspect/极性分离**，证明其向量质量更高。")

    col1, col2, col3 = st.columns(3)

    # 🌟 关键修复：将图形生成和显示合并到 with 块内

    # AnglE-ALCE (您的创新)
    with col1:
        st.markdown("##### 1. AnglE-ALCE (您的创新)")
        fig_alce = apply_tsne_and_plot(df_angle_alce, vectors_angle_alce, "AnglE-ALCE")
        st.plotly_chart(fig_alce, use_container_width=True) # 放在 with 块内

    # Dot-Product (基线对比)
    with col2:
        st.markdown("##### 2. Dot-Product (传统对比学习)")
        fig_dot = apply_tsne_and_plot(df_dot_product, vectors_dot_product, "Dot-Product")
        st.plotly_chart(fig_dot, use_container_width=True) # 放在 with 块内
        
    # Baseline BERT-CLS (原始模型)
    with col3:
        st.markdown("##### 3. Baseline BERT-CLS (原始基线)")
        fig_bert = apply_tsne_and_plot(df_bert, vectors_bert, "BERT-CLS")
        st.plotly_chart(fig_bert, use_container_width=True) # 放在 with 块内

    # --- 6. 选中的数据表格 ---
    st.header("选中的评估样本")
    st.markdown("以下是用于上述可视化和特征提取的评论样本。")
    st.dataframe(final_df[['review', 'label_aspect', 'label_polarity']], 
                 column_config={"review": "评论内容", "label_aspect": "所属方面", "label_polarity": "情感极性"},
                 use_container_width=True,
                 hide_index=True)


if __name__ == "__main__":
    main_app()


# streamlit run D:\WorkSpace\AnglE_yj\visible\visualize_space.py