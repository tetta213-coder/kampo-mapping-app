import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

# ページの設定
st.set_page_config(page_title="漢方マッピング・ラボ", layout="wide")

st.title("🌿 漢方処方インタラクティブ・マップ")
st.write("生薬構成と適応症フラグから、処方の『距離』を可視化します。")

# 1. データの読み込み
@st.cache_data
def load_data():
    return pd.read_csv("dose_shouyaku_standardized.csv")

df = load_data()

# 2. サイドバーの設定
st.sidebar.header("🔍 解析設定")
symptom_options = ["すべて", "GI (胃腸)", "Resp (呼吸器)", "Pain (痛み)", "Mental (精神)"]
selected_symp = st.sidebar.selectbox("ターゲット症状で絞り込む", symptom_options)
perplexity = st.sidebar.slider("Perplexity (近傍の意識度)", 5, 50, 25)
seed = st.sidebar.number_input("乱数シード (配置を変える)", value=42)

# 3. データのフィルタリング
plot_df = df.copy()
if selected_symp != "すべて":
    col_name = f"Flag_{selected_symp.split(' ')[0]}"
    if col_name in plot_df.columns:
        plot_df = plot_df[plot_df[col_name] > 0]

# 4. 体力判定列の作成（確実に存在する列から作成）
def judge_strength(row):
    if row['Flag_Strength_High'] > 0: return "実証"
    if row['Flag_Strength_Mid'] > 0: return "中間"
    if row['Flag_Strength_Low'] > 0: return "虚証"
    return "不明"

plot_df['体力'] = plot_df.apply(judge_strength, axis=1)

# 5. t-SNEの計算
# 数値列だけを抽出し、欠損値を埋める
numeric_data = plot_df.select_dtypes(include=[np.number]).drop(columns=['No'], errors='ignore').fillna(0)

# 重複行を避けるために極小のノイズを追加
numeric_data = numeric_data + np.random.normal(0, 1e-10, numeric_data.shape)

tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init='pca', learning_rate='auto')
res = tsne.fit_transform(numeric_data)
plot_df['x'] = res[:, 0]
plot_df['y'] = res[:, 1]

# 6. Plotlyによる描画
# color_discrete_map の値を安全な16進数カラーコードに変更
fig = px.scatter(
    plot_df, x='x', y='y',
    text='formula',
    color='体力',
    color_discrete_map={
        "実証": "#000000",   # 黒
        "中間": "#808080",   # 灰色
        "虚証": "#FFFFFF",   # 白
        "不明": "#D3D3D3"    # 薄い灰色
    },
    hover_name='formula',
    hover_data={
        'x': False, 
        'y': False, 
        'formula': False, # ★ここをFalseにすることで "formula=..." の表示が消えます
        '体力': True
    },
    height=700
)

# マーカーとテキストの調整（白丸が消えないように外枠を黒くする）
fig.update_traces(
    textposition='top center', 
    marker=dict(size=12, line=dict(width=1, color='black')),
    textfont=dict(family="HiraKakuPro-W3")
)

st.plotly_chart(fig, use_container_width=True)
st.info(f"現在の表示数: {len(plot_df)} 件 / マウスを点に当てると詳細が表示されます。")
