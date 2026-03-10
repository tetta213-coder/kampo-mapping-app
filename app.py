import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# 1. ページ基本設定
st.set_page_config(page_title="漢方マッピング・ラボ", layout="wide")

# --- ライトモード固定CSS ---
st.markdown("""
    <style>
    .stApp { background-color: white !important; color: #31333F !important; }
    [data-testid="stHeader"] { background-color: white !important; }
    [data-testid="stSidebar"] { background-color: #f0f2f6 !important; }
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp span { color: #31333F !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌿 証空間の出会い：患者と処方のマッチング")

# 2. データの読み込み (24次元統合データを使用)
@st.cache_data
def load_integrated_data():
    # EQ列まであるCSVを読み込む
    df = pd.read_csv("kampo_yakuno_integrated.csv")
    return df

df_full = load_integrated_data()

# --- 3. サイドバー：患者の「証」入力 ---
st.sidebar.header("👤 患者の病態入力")

# A. 主要10指標 (スライダー)
st.sidebar.subheader("基本の10指標")
sho_input = {}
sho_names = ['虚実', '寒', '熱', '気虚', '気鬱', '気逆', '血虚', '瘀血', '水毒', '腎虚']
for name in sho_names:
    sho_input[name] = st.sidebar.slider(f"{name}", 0.0, 1.0, 0.3)

# B. 個別14症状 (ラジオボタン/セレクトボックス)
st.sidebar.subheader("特定の随伴症状")
raw_input = {}
symptom_list = [
    "安心鎮静 (不眠・不安)", "認知知能 (物忘れ)", "鎮痙 (足のつり)", "眼精疲労", 
    "清頭目 (のぼせ・頭痛)", "排膿 (にきび)", "解毒 (かゆみ)", "疣贅 (いぼ)", 
    "制吐・鎮嘔", "瀉下 (便秘)", "黄疸", "安胎", "通乳"
]
for sym in symptom_list:
    raw_input[sym] = st.sidebar.radio(f"{sym}", ["なし", "あり"], index=0, horizontal=True)

# --- 4. 24次元患者ベクトルの生成 ---
def create_patient_vec(sho, raw):
    p = {
        "補気": sho['気虚'] + (sho['腎虚'] * 0.3),
        "理気": sho['気鬱'],
        "降気": sho['気逆'],
        "補血": sho['血虚'] + (sho['腎虚'] * 0.4),
        "駆瘀血": sho['瘀血'],
        "止血": 0, "利水": sho['水毒'],
        "潤水": (sho['腎虚'] * 0.3),
        "温": sho['寒'], "清": sho['熱']
    }
    # ラジオボタンの「あり」を 0.8 として加算
    others = ["安心鎮静", "認知知能", "鎮痙", "眼精疲労", "清頭目", "排膿", "解毒", "疣贅", "制吐", "鎮嘔", "瀉下", "黄疸", "安胎", "通乳"]
    for i, name in enumerate(others):
        p[name] = 0.8 if raw.get(symptom_list[i]) == "あり" else 0.0
    
    # 代理ロジック
    if p["降気"] < 0.3 and sho['気鬱'] > 0.5 and sho['熱'] > 0.5:
        p["降気"] = (sho['気鬱'] + sho['熱']) / 2
        
    return np.array(list(p.values()))

patient_vec = create_patient_vec(sho_input, raw_input)

# --- 5. 地図の計算 (24次元薬能スコアを使用) ---
# 右端24列が薬能軸
yakuno_data = df_full.iloc[:, -24:].fillna(0)

# t-SNE計算 (座標固定のためseedを固定)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
coords = tsne.fit_transform(yakuno_data)
df_full['x'], df_full['y'] = coords[:, 0], coords[:, 1]

# --- 6. マッチング計算 (★の位置を特定) ---
similarities = cosine_similarity([patient_vec], yakuno_data.values)[0]
df_full['similarity'] = similarities

# 上位3件の座標の重心を★の位置にする
top_3 = df_full.sort_values('similarity', ascending=False).head(3)
star_x = top_3['x'].mean()
star_y = top_3['y'].mean()

# --- 7. 地図描画 ---
fig = px.scatter(
    df_full, x='x', y='y', text='formula',
    color='similarity', color_continuous_scale='Viridis',
    hover_name='formula', height=800,
    title="漢方薬能空間：スライダーを動かすと★（患者）が移動します"
)

# 患者の現在地（★）を追加
fig.add_trace(go.Scatter(
    x=[star_x], y=[star_y],
    mode='markers+text',
    marker=dict(symbol='star', size=25, color='red', line=dict(width=2, color='white')),
    text=["★ あなたの現在地"],
    textposition="top center",
    name="患者の証"
))

fig.update_traces(textposition='top center', marker=dict(size=10))
fig.update_layout(plot_bgcolor='white', xaxis=dict(visible=False), yaxis=dict(visible=False))

st.plotly_chart(fig, use_container_width=True)

# 推薦処方の表示
st.subheader("🌟 あなたに推奨される処方トップ3")
cols = st.columns(3)
for i, (idx, row) in enumerate(top_3.iterrows()):
    cols[i].metric(f"{i+1}. {row['formula']}", f"一致度: {row['similarity']:.2%}")
