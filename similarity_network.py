# similarity_network.py
# Distance / similarity network (5 factors, FINAL)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# パス設定（スクリプト位置基準）
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "analysis_case_factor_means.csv")
OUT_IMG = os.path.join(BASE_DIR, "case_similarity_network.png")

# ==================================================
# データ読み込み & 検証
# ==================================================
if not os.path.isfile(INPUT_CSV):
    raise FileNotFoundError(f"❌ 入力CSVが見つかりません: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

FACTORS = ["Land", "Labor", "Culture", "Political", "Economic"]
required = {"Case"} | set(FACTORS)
if not required.issubset(df.columns):
    raise ValueError(f"❌ 必須列が不足: {required - set(df.columns)}")

case_means = df.set_index("Case")[FACTORS]
cases = case_means.index.tolist()
X = case_means.values

# ==================================================
# 類似度計算
# ==================================================
# ユークリッド（構造）
euclid_dist = squareform(pdist(X, metric="euclidean"))
euclid_max = euclid_dist.max() if euclid_dist.max() > 0 else 1.0
euclid_sim = 1 - euclid_dist / euclid_max  # 0–1

# コサイン（方向）
cos_sim = cosine_similarity(X)  # -1–1 → 実際は 0–1

# 総合類似度
final_sim = 0.5 * euclid_sim + 0.5 * cos_sim
sim_df = pd.DataFrame(final_sim, index=cases, columns=cases)

# ==================================================
# ネットワーク構築
# ==================================================
G = nx.Graph()
for c in cases:
    G.add_node(c)

# 閾値（調整可：0.75 ≒ あなたの結果で「意味のある近さ」）
THRESHOLD = 0.75

for i, c1 in enumerate(cases):
    for j, c2 in enumerate(cases):
        if j <= i:
            continue
        w = sim_df.loc[c1, c2]
        if w >= THRESHOLD:
            G.add_edge(c1, c2, weight=w)

# ==================================================
# 描画（Force-directed）
# ==================================================
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=42, weight="weight")

# ノード
nx.draw_networkx_nodes(
    G, pos,
    node_size=2000,
    node_color="#EEEEEE",
    edgecolors="black"
)

# エッジ（太さ = 類似度）
edges = G.edges(data=True)
widths = [2 + 6 * (d["weight"] - THRESHOLD) for (_, _, d) in edges]

nx.draw_networkx_edges(
    G, pos,
    width=widths,
    alpha=0.8
)

# ラベル
nx.draw_networkx_labels(
    G, pos,
    font_size=9
)

plt.title("Case Similarity Network (5 Factors, threshold ≥ 0.75)")
plt.axis("off")
plt.tight_layout()

plt.savefig(OUT_IMG, dpi=150)
plt.show()

print(f"✅ 距離ネットワーク図を保存しました: {OUT_IMG}")