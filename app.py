import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

# 1. ページ基本設定
st.set_page_config(page_title="漢方マッピング・ラボ", layout="wide")

st.title("🌿 漢方処方インタラクティブ・マップ")
st.write("生薬構成と適応症フラグから、処方の『距離』を可視化します。")

# 2. データの読み込み
@st.cache_data
def load_data():
    df = pd.read_csv("dose_shouyaku_standardized.csv")
    # ホバー時の "formula=" という表示を消すための究極の対策
    # 列名自体を「スペース1つ（" "）」に変更します
    df = df.rename(columns={'formula': ' '})
    return df

df = load_data()

# 3. サイドバーの設定
st.sidebar.header("🔍 解析設定")

# 【NGフィルタリング】
ng_filter = st.sidebar.radio(
    "NG処方の扱い",
    ["すべて表示", "NG処方を除外"],
    index=1
)

# 症状フィルター
symptom_options = ["すべて", "GI (胃腸)", "Resp (呼吸器)", "Pain (痛み)", "Mental (精神)"]
selected_symp = st.sidebar.selectbox("ターゲット症状で絞り込む", symptom_options)

# t-SNEパラメータ
perplexity = st.sidebar.slider("Perplexity (近傍の意識度)", 5, 50, 25)
seed = st.sidebar.number_input("乱数シード (配置を変える)", value=42)

# --- 4. データのフィルタリング（t-SNE計算の前に実行） ---
plot_df = df.copy()

# NG除外の実行
if ng_filter == "NG処方を除外":
    # NG列が 1 以外のもの（0 またはそれに近い値）だけを残す
    plot_df = plot_df[plot_df['NG'] < 0.5]

# 症状フィルターの実行
if selected_symp != "すべて":
    col_name = f"Flag_{selected_symp.split(' ')[0]}"
    if col_name in plot_df.columns:
        plot_df = plot_df[plot_df[col_name] > 0]

# 表示対象が少なすぎるとエラーになるための安全策
if len(plot_df) < perplexity:
    st.error(f"データ数が少なすぎます（現在 {len(plot_df)} 件）。Perplexityを下げてください。")
    st.stop()

# --- 5. 体力判定とt-SNE計算 ---
def judge_strength(row):
    if row['Flag_Strength_High'] > 0: return "実証"
    if row['Flag_Strength_Mid'] > 0: return "中間"
    if row['Flag_Strength_Low'] > 0: return "虚証"
    return "不明"

plot_df['体力'] = plot_df.apply(judge_strength, axis=1)

# 数値データのみを抽出
numeric_data = plot_df.select_dtypes(include=[np.number]).drop(columns=['No', 'NG'], errors='ignore').fillna(0)
# 重複回避のノイズ
numeric_data = numeric_data + np.random.normal(0, 1e-10, numeric_data.shape)

tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init='pca', learning_rate='auto')
res = tsne.fit_transform(numeric_data)
plot_df['x'] = res[:, 0]
plot_df['y'] = res[:, 1]

# --- 6. Plotlyによる描画 ---
# hover_nameに「スペース1つ」の列（元の処方名）を指定します
fig = px.scatter(
    plot_df, x='x', y='y',
    text=' ', 
    color='体力',
    color_discrete_map={"実証": "#000000", "中間": "#808080", "虚証": "#FFFFFF", "不明": "#D3D3D3"},
    hover_name=' ',
    hover_data={'x': False, 'y': False, ' ': False, '体力': True}
)

fig.update_traces(
    textposition='top center', 
    marker=dict(size=12, line=dict(width=1, color='black')),
    # ホバーの余計な箱を消す設定
    hovertemplate="<b>%{hovertext}</b><br>体力: %{customdata[0]}<extra></extra>",
    customdata=np.stack([plot_df['体力']], axis=-1)
)

st.plotly_chart(fig, use_container_width=True)
st.info(f"現在の表示処方数: {len(plot_df)} 件")
