import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

# 1. ページ基本設定
st.set_page_config(page_title="漢方マッピング・ラボ", layout="wide")

st.title("🌿 漢方処方知恵の地図")
st.write("生薬の組み合わせから、処方同士の「近さ」を AI が計算して地図にしました。")

# 2. データの読み込み
@st.cache_data
def load_data():
    df = pd.read_csv("dose_shouyaku_standardized.csv")
    # ラベルを綺麗にするために内部的に列名をリネーム
    df = df.rename(columns={'formula': '処方名'})
    return df

df = load_data()

# 3. サイドバーの設定（言葉を分かりやすく修正）
st.sidebar.header("🛠 マップの表示調整")

# --- NGフィルタリング ---
ng_filter = st.sidebar.radio(
    "1. データの信頼性",
    ["すべての処方を表示", "信頼できるデータのみ表示"],
    index=1,
    help="解析にノイズが混じるのを防ぐため、一部の不完全なデータ（NGフラグ付き）を隠すことができます。"
)

# --- 症状フィルター ---
symptom_dict = {
    "すべて": "すべて",
    "GI (胃腸)": "Flag_GI",
    "Resp (呼吸器)": "Flag_Resp",
    "Pain (痛み)": "Flag_Pain",
    "Mental (精神)": "Flag_Mental"
}
selected_label = st.sidebar.selectbox(
    "2. 注目したい症状",
    list(symptom_dict.keys()),
    help="特定の症状に効く処方だけに絞り込んで、その中での分布を詳しく見ることができます。"
)

# --- Perplexity（詳細度の翻訳） ---
perplexity = st.sidebar.slider(
    "3. 地図の描き込み度 (Perplexity)",
    min_value=5,
    max_value=50,
    value=25,
    help="【小さい値】近所の処方同士の集まり（クラスター）を重視します。\n【大きい値】地図全体のバランス（大局的な位置関係）を重視します。"
)

# --- 乱数シード（配置パターンの翻訳） ---
seed = st.sidebar.number_input(
    "4. 配置のパターン (Seed)",
    value=42,
    help="AIが計算するたびに、地図が回転したり反転したりします。中身の関係性は変わりませんが、見え方を変えたい時に数字を変えてください。"
)

# --- 4. データのフィルタリング ---
plot_df = df.copy()

if ng_filter == "信頼できるデータのみ表示":
    plot_df['NG'] = plot_df['NG'].astype(float)
    plot_df = plot_df[plot_df['NG'] == 0]

if selected_label != "すべて":
    col_name = symptom_dict[selected_label]
    if col_name in plot_df.columns:
        plot_df = plot_df[plot_df[col_name] > 0]

# --- 5. 体力判定とt-SNE計算 ---
def judge_strength(row):
    if row['Flag_Strength_High'] > 0: return "実証（体力あり）"
    if row['Flag_Strength_Mid'] > 0: return "中間"
    if row['Flag_Strength_Low'] > 0: return "虚証（体力控えめ）"
    return "不明"

plot_df['証（体力）'] = plot_df.apply(judge_strength, axis=1)

numeric_data = plot_df.select_dtypes(include=[np.number]).drop(columns=['No', 'NG'], errors='ignore').fillna(0)
numeric_data = numeric_data + np.random.normal(0, 1e-10, numeric_data.shape)

tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init='pca', learning_rate='auto')
res = tsne.fit_transform(numeric_data)
plot_df['x'] = res[:, 0]
plot_df['y'] = res[:, 1]

# --- 6. Plotlyによる描画 ---
fig = px.scatter(
    plot_df, x='x', y='y',
    text='処方名', 
    color='証（体力）',
    color_discrete_map={
        "実証（体力あり）": "#000000",
        "中間": "#808080",
        "虚証（体力控えめ）": "#FFFFFF",
        "不明": "#D3D3D3"
    },
    hover_name='処方名',
    # hover_data を使わずに hovertemplate で完全に制御
    custom_data=['証（体力）']
)

fig.update_traces(
    textposition='top center', 
    marker=dict(size=12, line=dict(width=1, color='black')),
    # ホバーから「列名=値」を完全に消去
    hovertemplate="<b>%{hovertext}</b><br>%{customdata[0]}<extra></extra>",
    textfont=dict(family="HiraKakuPro-W3")
)

st.plotly_chart(fig, use_container_width=True)
st.info(f"💡 現在 {len(plot_df)} 件の処方を表示中。マウスを点に当てると詳細が見えます。")
