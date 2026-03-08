import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

# 1. ページ基本設定
st.set_page_config(page_title="漢方マッピング・ラボ", layout="wide")

st.title("🌿 漢方処方知恵の地図")
st.write("生薬の組み合わせから、処方同士の「近さ」を可視化しました。")

# 2. データの読み込み
@st.cache_data
def load_data():
    df = pd.read_csv("dose_shouyaku_standardized.csv")
    return df

df = load_data()

# 3. サイドバーの設定
st.sidebar.header("🛠 マップの表示調整")

# --- 1. 保険適用によるフィルタリング ---
ng_filter = st.sidebar.radio(
    "1. 解析対象の範囲",
    ["すべての処方（保険外を含む）", "保険適用148処方のみ"],
    index=1,
    help="NG列が0のものを『保険適用148処方』、1のものを『保険外』として扱います。"
)

# 症状フィルター
symptom_dict = {"すべて": "すべて", "GI (胃腸)": "Flag_GI", "Resp (呼吸器)": "Flag_Resp", "Pain (痛み)": "Flag_Pain", "Mental (精神)": "Flag_Mental"}
selected_label = st.sidebar.selectbox("2. 注目したい症状", list(symptom_dict.keys()))

# パラメータ
perplexity = st.sidebar.slider("3. 地図の描き込み度 (Perplexity)", 5, 50, 25)
seed = st.sidebar.number_input("4. 配置のパターン (Seed)", value=42)

# --- 4. 【最重要】データのフィルタリング処理 ---
plot_df = df.copy()

# NG列を強制的に「数値」に変換し、不明な値は1（除外対象）にする
plot_df['NG'] = pd.to_numeric(plot_df['NG'], errors='coerce').fillna(1)

if ng_filter == "保険適用148処方のみ":
    # 0.5未満を「0」と判定する（浮動小数点の誤差や微細な数値対策）
    plot_df = plot_df[plot_df['NG'] < 0.5]
    
# 症状フィルターの適用
if selected_label != "すべて":
    col_name = symptom_dict[selected_label]
    if col_name in plot_df.columns:
        plot_df = plot_df[plot_df[col_name] > 0]

# --- 5. サイドバーに対象数を表示（デバッグ用） ---
st.sidebar.markdown(f"### 📊 現在の対象: **{len(plot_df)}** 処方")
if len(plot_df) == 0:
    st.error("表示対象が0件です。設定を見直してください。")
    st.stop()

# --- 6. 体力判定とt-SNE計算 ---
def judge_strength(row):
    if row['Flag_Strength_High'] > 0: return "実証（体力あり）"
    if row['Flag_Strength_Mid'] > 0: return "中間"
    if row['Flag_Strength_Low'] > 0: return "虚証（体力控えめ）"
    return "不明"

plot_df['証（体力）'] = plot_df.apply(judge_strength, axis=1)

# 数値列だけを取り出して計算
numeric_data = plot_df.select_dtypes(include=[np.number]).drop(columns=['No', 'NG'], errors='ignore').fillna(0)
numeric_data = numeric_data + np.random.normal(0, 1e-10, numeric_data.shape)

tsne = TSNE(n_components=2, perplexity=min(perplexity, len(plot_df)-1), random_state=seed, init='pca', learning_rate='auto')
res = tsne.fit_transform(numeric_data)
plot_df['x'] = res[:, 0]
plot_df['y'] = res[:, 1]

# --- 7. Plotlyによる描画（formula= を消す究極設定） ---
fig = px.scatter(
    plot_df, x='x', y='y',
    text='formula', 
    color='証（体力）',
    color_discrete_map={"実証（体力あり）": "#000000", "中間": "#808080", "虚証（体力控えめ）": "#FFFFFF", "不明": "#D3D3D3"},
    # hover_data自体を空にして、hovertemplateで再構築する
    hover_data={'x': False, 'y': False, 'formula': False, '証（体力）': False}
)

fig.update_traces(
    textposition='top center', 
    marker=dict(size=12, line=dict(width=1, color='black')),
    # customdataにformulaと証を渡して、表示を完全制御
    customdata=np.stack([plot_df['formula'], plot_df['証（体力）']], axis=-1),
    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
    textfont=dict(family="HiraKakuPro-W3")
)

st.plotly_chart(fig, use_container_width=True)
