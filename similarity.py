# similarity.py — Case similarity analysis (5 factors, FINAL FIXED)
# Input: analysis_case_factor_means.csv

import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# パス設定（このスクリプトの場所基準）
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "analysis_case_factor_means.csv")

# ==================================================
# データ読み込み
# ==================================================
if not os.path.isfile(INPUT_CSV):
    raise FileNotFoundError(f"❌ 入力CSVが見つかりません: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

FACTORS = ["Land", "Labor", "Culture", "Political", "Economic"]
required = {"Case"} | set(FACTORS)
if not required.issubset(df.columns):
    raise ValueError(
        f"❌ 必須列が不足しています\n"
        f"必要: {required}\n"
        f"実際: {set(df.columns)}"
    )

case_means = df.set_index("Case")[FACTORS]
cases = case_means.index.tolist()
X = case_means.values

# ==================================================
# 1) ユークリッド距離 → 類似度（0–100）
#    ※ 構造差（量的な離れ具合）を反映
# ==================================================
euclid_dist = squareform(pdist(X, metric="euclidean"))
euclid_max = euclid_dist.max() if euclid_dist.max() > 0 else 1.0
euclid_sim = 100 * (1 - euclid_dist / euclid_max)

# ==================================================
# 2) コサイン類似度（0–100）
#    ※ 因子配分の「方向」を反映
# ==================================================
cos_sim = cosine_similarity(X) * 100

# ==================================================
# 3) 総合類似度（同重み）
# ==================================================
final_sim = 0.5 * euclid_sim + 0.5 * cos_sim
sim_df = pd.DataFrame(final_sim, index=cases, columns=cases).round(2)

print("\n=== Case Similarity Matrix (0–100) ===")
print(sim_df.to_string())

# ==================================================
# 4) Ainu（Edo / Meiji）の近接事例
# ==================================================
def top_similar(case_name, sim, n=5):
    if case_name not in sim.index:
        raise ValueError(f"Case not found: {case_name}")
    return (
        sim.loc[case_name]
        .drop(case_name)
        .sort_values(ascending=False)
        .head(n)
    )

EDO   = "Ainu (Edo Period - Basho Ukeoi)"
MEIJI = "Ainu (Meiji Period - Former Aborigine Law)"

print("\n=== Ainu (Edo) に類似する事例 ===")
print(top_similar(EDO, sim_df))

print("\n=== Ainu (Meiji) に類似する事例 ===")
print(top_similar(MEIJI, sim_df))