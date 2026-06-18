# =========================================================
# mds_revised.py  （修正版）
# 主な変更点:
#   1. clusters定義をデンドログラムと旧図に合わせて修正
#   2. Ainu(Edo)を楕円から除外（境界事例として独立配置）
#   3. 楕円scaleをクラスターごとに最適化
#   4. ラベルオフセットを拡大して線との重複を解消
#   5. Ainu事例にダイヤモンドマーカーを使用
#   6. Ainu間の変化を示す線を明確化
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from matplotlib.patches import Ellipse

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "IPAexGothic", "IPA Gothic", "Noto Sans CJK JP",
    "Hiragino Sans", "Yu Gothic", "Meiryo", "MS Gothic"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11

# =========================================================
# 1. 基本設定
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "analysis_case_factor_means.csv")

FACTORS = ["Land", "Labor", "Culture", "Political", "Economic"]

EDO_KEY   = "Ainu (Edo Period - Basho Ukeoi)"
MEIJI_KEY = "Ainu (Meiji Period - Former Aborigine Law)"

# =========================================================
# 2. データ読み込み
# =========================================================

df = pd.read_csv(INPUT_CSV)
df["Case"] = df["Case"].astype(str).str.strip()

X     = df.set_index("Case")[FACTORS]
cases = X.index.tolist()
Xv    = X.values

EDO   = EDO_KEY
MEIJI = MEIJI_KEY

# =========================================================
# 3. クラスター定義（修正版）
#    デンドログラム（Ward法 k=3）および旧図の楕円配置に基づく
#    Ainu(Edo)は境界事例のため楕円から除外
# =========================================================

clusters = {
    "搾取型 / Exploitation-type": [
        "Bengal (British India)",
        "Indonesia (Dutch East Indies)",
    ],
    "入植同化型 / Settlement-type": [
        "Maori (New Zealand)",
        "Aboriginal Australians",
        "Native American (US)",
        "Ireland",
        MEIJI_KEY,
    ],
    "東アジア帝国型 / East-Asian Imperial": [
        "Taiwan (Japanese Rule)",
        "Korea (Japanese Rule)",
        "Ryukyu (Okinawa)",
    ],
}

# 楕円の線種
cluster_linestyles = {
    "搾取型 / Exploitation-type":          "--",
    "入植同化型 / Settlement-type":         "-.",
    "東アジア帝国型 / East-Asian Imperial": ":",
}

# 楕円のscale（クラスターごとに調整）
cluster_scales = {
    "搾取型 / Exploitation-type":          2.2,   # 2事例
    "入植同化型 / Settlement-type":         2.8,   # 5事例・広がり大
    "東アジア帝国型 / East-Asian Imperial": 2.0,   # 3事例・密集
}

# =========================================================
# 4. 日英ラベル
# =========================================================

labels_bilingual = {
    "Ainu (Edo Period - Basho Ukeoi)":           "アイヌ（場所請負制期）\nAinu (Edo)",
    "Ainu (Meiji Period - Former Aborigine Law)": "アイヌ（北海道入植植民期）\nAinu (Meiji)",
    "Bengal (British India)":                    "ベンガル（英領）\nBengal",
    "Indonesia (Dutch East Indies)":             "ジャワ（蘭領東インド）\nIndonesia (Java)",
    "Korea (Japanese Rule)":                     "朝鮮（日本統治）\nKorea",
    "Native American (US)":                      "米国先住民\nNative Americans",
    "Taiwan (Japanese Rule)":                    "台湾（日本統治）\nTaiwan",
    "Ireland":                                   "アイルランド\nIreland",
    "Maori (New Zealand)":                       "マオリ\nMaori (NZ)",
    "Aboriginal Australians":                    "豪州アボリジニ\nAboriginal Aus.",
    "Ryukyu (Okinawa)":                          "琉球（沖縄）\nRyukyu",
}

# =========================================================
# 5. 類似度・非類似度計算（既存コードと同一）
# =========================================================

euclid_dist  = squareform(pdist(Xv, metric="euclidean"))
euclid_sim   = 100 * (1 - euclid_dist / euclid_dist.max())
cos_sim      = cosine_similarity(Xv) * 100
final_sim    = 0.5 * euclid_sim + 0.5 * cos_sim
dissimilarity = 100 - final_sim

# =========================================================
# 6. MDS
# =========================================================

mds    = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
coords = mds.fit_transform(dissimilarity)
mds_df = pd.DataFrame(coords, index=cases, columns=["Dim1", "Dim2"])

# =========================================================
# 7. 楕円描画関数（scaleを引数で受け取る）
# =========================================================

def draw_cluster_ellipse(ax, df, case_list, linestyle, scale=2.0, min_size=0.8):
    pts    = df.loc[case_list][["Dim1", "Dim2"]].values
    center = pts.mean(axis=0)

    if len(pts) == 2:
        diff   = pts[1] - pts[0]
        dist   = np.linalg.norm(diff)
        width  = max(dist * scale, min_size)
        height = max(dist * 0.45, min_size)
        angle  = np.degrees(np.arctan2(diff[1], diff[0]))
    else:
        cov  = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals  = vals[order]
        vecs  = vecs[:, order]
        width  = max(scale * np.sqrt(vals[0]), min_size)
        height = max(scale * np.sqrt(vals[1]), min_size)
        angle  = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    ellipse = Ellipse(
        xy        = center,
        width     = width * 2,   # Ellipseはwidth=直径
        height    = height * 2,
        angle     = angle,
        facecolor = "none",
        edgecolor = "black",
        linestyle = linestyle,
        linewidth = 1.5
    )
    ax.add_patch(ellipse)

# =========================================================
# 8. 描画
# =========================================================

fig, ax = plt.subplots(figsize=(10, 8))

ax.axhline(0, color="gray", lw=0.5, zorder=1)
ax.axvline(0, color="gray", lw=0.5, zorder=1)

# 楕円描画
for cname, case_list in clusters.items():
    draw_cluster_ellipse(
        ax, mds_df, case_list,
        linestyle = cluster_linestyles[cname],
        scale     = cluster_scales[cname],
        min_size  = 1.0
    )

# =========================================================
# 9. 通常マーカー（Ainu2事例を除く）
# =========================================================

non_ainu = [c for c in cases if c not in {EDO, MEIJI}]
ax.scatter(
    mds_df.loc[non_ainu, "Dim1"],
    mds_df.loc[non_ainu, "Dim2"],
    s=55, facecolors="white", edgecolors="black", zorder=4
)

# Ainuはダイヤモンドマーカー
for key in [EDO, MEIJI]:
    x, y = mds_df.loc[key, ["Dim1", "Dim2"]]
    ax.scatter(x, y, s=90, marker="D",
               facecolors="white", edgecolors="black",
               linewidths=1.5, zorder=5)

# アイヌ2事例間の線（変化を示す）
x_edo,   y_edo   = mds_df.loc[EDO,   ["Dim1", "Dim2"]]
x_meiji, y_meiji = mds_df.loc[MEIJI, ["Dim1", "Dim2"]]
ax.plot([x_edo, x_meiji], [y_edo, y_meiji],
        color="black", lw=1.2, ls="--", zorder=3, alpha=0.7)

# =========================================================
# 10. ラベル描画（通常事例）
#     オフセットを0.18に拡大してラインと重複しないよう調整
# =========================================================

cx, cy = mds_df["Dim1"].mean(), mds_df["Dim2"].mean()
OFFSET = 0.18   # 旧コードの0.09から拡大

for case in non_ainu:
    x, y   = mds_df.loc[case, ["Dim1", "Dim2"]]
    label  = labels_bilingual[case]
    dx, dy = x - cx, y - cy
    norm   = np.sqrt(dx**2 + dy**2) + 1e-6
    ox     = OFFSET * dx / norm
    oy     = OFFSET * dy / norm

    ha = "left"  if ox >= 0 else "right"
    va = "bottom" if oy >= 0 else "top"

    # リーダーライン（点から少し離してテキスト開始）
    ax.plot([x, x + ox * 0.6], [y, y + oy * 0.6],
            color="black", lw=0.5, alpha=0.6, zorder=3)

    ax.text(x + ox, y + oy, label,
            fontsize=8.5, ha=ha, va=va, zorder=5,
            linespacing=1.3)

# =========================================================
# 11. アイヌ2事例ラベル（専用・大きめフォント）
#     リーダーラインとテキストの間に十分な余白を確保
# =========================================================

# Ainu(Edo) → 右上
ox_edo, oy_edo = 1.8, 1.2
ax.plot([x_edo, x_edo + ox_edo * 0.6],
        [y_edo, y_edo + oy_edo * 0.6],
        color="black", lw=0.9, zorder=3)
ax.text(x_edo + ox_edo, y_edo + oy_edo,
        labels_bilingual[EDO],
        fontsize=10, fontweight="bold",
        ha="left", va="bottom", linespacing=1.3, zorder=6)

# Ainu(Meiji) → 右下
ox_meiji, oy_meiji = 1.8, -1.2
ax.plot([x_meiji, x_meiji + ox_meiji * 0.6],
        [y_meiji, y_meiji + oy_meiji * 0.6],
        color="black", lw=0.9, zorder=3)
ax.text(x_meiji + ox_meiji, y_meiji + oy_meiji,
        labels_bilingual[MEIJI],
        fontsize=10, fontweight="bold",
        ha="left", va="top", linespacing=1.3, zorder=6)

# =========================================================
# 12. 凡例
# =========================================================

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], ls="--",  color="black", lw=1.5,
           label="搾取型 / Exploitation-type"),
    Line2D([0], [0], ls="-.",  color="black", lw=1.5,
           label="入植同化型 / Settlement-type"),
    Line2D([0], [0], ls=":",   color="black", lw=1.5,
           label="東アジア帝国型 / East-Asian Imperial"),
]
ax.legend(handles=legend_elements, loc="lower right",
          fontsize=8.5, framealpha=0.9)

# =========================================================
# 13. 軸・タイトル
# =========================================================

ax.set_xlabel("次元1 / Dimension 1（類似度空間）", fontsize=10)
ax.set_ylabel("次元2 / Dimension 2（類似度空間）", fontsize=10)
ax.set_title(
    "植民地統治プロファイルのMDS配置図（５因子空間・類似度基準）\n"
    "MDS Plot of Colonial Governance Profiles (5-Factor Similarity Space)",
    fontsize=11
)

plt.tight_layout()
output_path = os.path.join(BASE_DIR, "mds_revised.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.close()
