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
    # 前回作成した標準化済みデータを読み込む
    return pd.read_csv("dose_shouyaku_standardized.csv")

df = load_data()

# 2. サイドバーの設定（操作パネル）
st.sidebar.header("🔍 解析設定")

# 症状フィルター
symptom_options = ["すべて", "GI (胃腸)", "Resp (呼吸器)", "Pain (痛み)", "Mental (精神)"]
selected_symp = st.sidebar.selectbox("ターゲット症状で絞り込む", symptom_options)

# t-SNEのパラメータ
perplexity = st.sidebar.slider("Perplexity (近傍の意識度)", 5, 50, 25)
seed = st.sidebar.number_input("乱数シード (配置を変える)", value=42)

# 3. データのフィルタリング
plot_df = df.copy()
if selected_symp != "すべて":
    # 選択された症状フラグがプラス（標準化後なので0より大きい）ものに絞る
    col_name = f"Flag_{selected_symp.split(' ')[0]}"
    plot_df = plot_df[plot_df[col_name] > 0]

st.sidebar.write(f"現在の対象処方数: {len(plot_df)} 件")

# 4. t-SNEの計算
# 数値列だけを抽出（No, formula, Indication_Textなどは除外）
numeric_data = plot_df.select_dtypes(include=[np.number]).drop(columns=['No'], errors='ignore')
# NaNがあれば0で埋める
numeric_data = numeric_data.fillna(0)

tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init='pca', learning_rate='auto')
res = tsne.fit_transform(numeric_data)

# プロット用データ作成
plot_df['x'] = res[:, 0]
plot_df['y'] = res[:, 1]
# 体力判定（色分け用）
plot_df['体力'] = plot_df['Flag_Strength_High'].apply(lambda x: "実証" if x > 0 else "その他")

# --- 修正後のコード（ここから） ---

# 4. 体力の判定ロジック（確実に存在する列を使用）
def judge_strength(row):
    if row['Flag_Strength_High'] > 0: return "実証"
    if row['Flag_Strength_Mid'] > 0: return "中間"
    if row['Flag_Strength_Low'] > 0: return "虚証"
    return "不明"

plot_df['体力'] = plot_df.apply(judge_strength, axis=1)

# 5. Plotlyによる描画
fig = px.scatter(
    plot_df, x='x', y='y',
    text='formula',
    color='体力',
    color_discrete_map={
        "実証": "black", 
        "中間": "grey", 
        "虚証": "white", 
        "不明": "transparent"
    },
    hover_name='formula',
    # 'Indication_Text' を削除し、存在する列だけにする
    hover_data={
        'x': False, 
        'y': False, 
        '体力': True,
        'Flag_GI': True,     # 消化器フラグなど、見たいものを追加
        'Flag_Resp': True    # 呼吸器フラグ
    },
    height=700
)

# 見た目の調整
fig.update_traces(
    textposition='top center', 
    marker=dict(size=12, line=dict(width=1, color='black'))
)
# --- 修正後のコード（ここまで） ---
fig.update_traces(textposition='top center', marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(dragmode='pan') # 初期設定を「パン（移動）」にする

st.plotly_chart(fig, use_container_width=True)

st.info("💡 マウスホイールでズーム、ドラッグで移動、点にホバーすると適応症が表示されます。")
