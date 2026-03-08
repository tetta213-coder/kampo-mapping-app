import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

st.set_page_config(page_title="漢方マッピング", layout="wide")

# 【対策】あえてキャッシュを使わずに毎回読み込む設定にします
def load_data_fresh():
    df = pd.read_csv("dose_shouyaku_standardized.csv")
    return df

df = load_data_fresh()

st.sidebar.header("🔍 設定")
ng_filter = st.sidebar.radio("NG処方の扱い", ["すべて表示", "NG処方を除外"], index=1)
perplexity = st.sidebar.slider("Perplexity", 5, 50, 25)
seed = st.sidebar.number_input("Seed", value=42)

# --- フィルタリング ---
plot_df = df.copy()

if ng_filter == "NG処方を除外":
    # 数値でも文字列でも対応できるように型変換してからフィルタリング
    plot_df['NG'] = plot_df['NG'].astype(float)
    plot_df = plot_df[plot_df['NG'] == 0]

# 【デバッグ用】本当にNGが消えているか画面で確認
st.write(f"現在のデータ件数: {len(plot_df)} 件 (NG=1の件数: {len(plot_df[plot_df['NG'] > 0])})")

# --- t-SNE ---
numeric_data = plot_df.select_dtypes(include=[np.number]).drop(columns=['No', 'NG'], errors='ignore').fillna(0)
numeric_data = numeric_data + np.random.normal(0, 1e-10, numeric_data.shape)

tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init='pca', learning_rate='auto')
res = tsne.fit_transform(numeric_data)
plot_df['x'] = res[:, 0]
plot_df['y'] = res[:, 1]

# --- 描画 (ラベル問題を力技で解決) ---
fig = px.scatter(
    plot_df, x='x', y='y',
    text='formula',
    color='Flag_Strength_High', # 代わりに生の色データを使用
    # formula という単語を出さないために labels で上書き
    labels={'formula': '', 'x': 't-SNE 1', 'y': 't-SNE 2'}
)

# ホバー設定を「処方名のみ」に極限までシンプル化
fig.update_traces(
    hovertemplate="<b>%{text}</b><extra></extra>",
    textposition='top center',
    marker=dict(size=10, line=dict(width=1, color='black'))
)

st.plotly_chart(fig, use_container_width=True)
