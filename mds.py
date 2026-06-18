import os, numpy as np, pandas as pd, platform, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

def set_japanese_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    candidates = {
        "Darwin":  ["Hiragino Sans","Hiragino Maru Gothic Pro"],
        "Windows": ["Meiryo","Yu Gothic","MS Gothic"],
        "Linux":   ["Noto Sans CJK JP","IPAexGothic","VL Gothic"],
    }
    for font in candidates.get(platform.system(), []) + ["DejaVu Sans"]:
        if font in available:
            rcParams["font.family"] = font; return
    rcParams["font.family"] = "DejaVu Sans"

set_japanese_font()
rcParams["axes.unicode_minus"] = False

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "analysis_case_factor_means.csv")
FACTORS   = ["Land","Labor","Culture","Political","Economic"]
EDO_KEY   = "Ainu (Edo Period - Basho Ukeoi)"
MEIJI_KEY = "Ainu (Meiji Period - Former Aborigine Law)"

# =========================================================
# k=4 クラスター定義
# =========================================================
clusters = {
    "入植同化型": {
        "cases":  [MEIJI_KEY, "Maori (New Zealand)", "Aboriginal Australians"],
        "ls": (0,(6,3)), "lw": 1.4, "color": "#909090",
        "scale": 1.7, "min_h": 5.0,
        "label": "Cluster 4（入植同化型 / Settlement-type）",
    },
    "移行複合型": {
        "cases":  [EDO_KEY, "Native American (US)", "Ireland"],
        "ls": (0,(10,3,2,3)), "lw": 1.4, "color": "#585858",
        "scale": 1.4, "min_h": 7.0,   # min_hで薄くなりすぎを防ぐ
        "label": "Cluster 3（移行複合型 / Transitional-type）",
    },
    "東アジア帝国型": {
        "cases":  ["Taiwan (Japanese Rule)","Korea (Japanese Rule)","Ryukyu (Okinawa)"],
        "ls": (0,(3,2,1,2)), "lw": 1.6, "color": "#303030",
        "scale": 1.7, "min_h": 5.0,
        "label": "Cluster 2（東アジア帝国型 / East-Asian Imperial）",
    },
    "搾取型": {
        "cases":  ["Bengal (British India)","Indonesia (Dutch East Indies)"],
        "ls": "--", "lw": 1.8, "color": "#080808",
        "scale": 1.5, "min_h": 4.0,
        "label": "Cluster 1（搾取型 / Exploitation-type）",
    },
}

labels_bilingual = {
    EDO_KEY:                         "アイヌ（場所請負制期）\nAinu (Edo)",
    MEIJI_KEY:                       "アイヌ（北海道入植植民期）\nAinu (Meiji)",
    "Bengal (British India)":        "ベンガル（英領）\nBengal",
    "Indonesia (Dutch East Indies)": "ジャワ（蘭領東インド）\nIndonesia (Java)",
    "Korea (Japanese Rule)":         "朝鮮（日本統治）\nKorea",
    "Native American (US)":          "米国先住民\nNative Americans",
    "Taiwan (Japanese Rule)":        "台湾（日本統治）\nTaiwan",
    "Ireland":                       "アイルランド\nIreland",
    "Maori (New Zealand)":           "マオリ\nMaori (NZ)",
    "Aboriginal Australians":        "豪州アボリジニ\nAboriginal Aus.",
    "Ryukyu (Okinawa)":              "琉球（沖縄）\nRyukyu",
}

# =========================================================
# データ読み込み・MDS
# =========================================================
df = pd.read_csv(INPUT_CSV); df["Case"] = df["Case"].str.strip()
X  = df.set_index("Case")[FACTORS]; Xv = X.values; cases = X.index.tolist()

euclid_dist   = squareform(pdist(Xv, metric="euclidean"))
euclid_sim    = 100*(1 - euclid_dist/euclid_dist.max())
cos_sim       = cosine_similarity(Xv)*100
dissimilarity = 100 - (0.5*euclid_sim + 0.5*cos_sim)

mds    = MDS(n_components=2, dissimilarity="precomputed",
             random_state=42, n_init=4)
coords = mds.fit_transform(dissimilarity)
mds_df = pd.DataFrame(coords, index=cases, columns=["Dim1","Dim2"])

# =========================================================
# 楕円描画（min_h: 短軸の最小半径）
# =========================================================
def draw_ellipse(ax, pts, color, ls, lw, scale, min_h):
    center = pts.mean(axis=0)
    if len(pts) == 2:
        diff   = pts[1] - pts[0]; dist = np.linalg.norm(diff)
        semi_a = max(dist * scale / 2, min_h)
        semi_b = max(dist * 0.25, min_h * 0.5)
        angle  = np.degrees(np.arctan2(diff[1], diff[0]))
    else:
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]; vals = vals[order]; vecs = vecs[:,order]
        semi_a = max(scale * np.sqrt(abs(vals[0])), min_h)
        semi_b = max(scale * np.sqrt(abs(vals[1])), min_h * 0.45)
        angle  = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    ax.add_patch(Ellipse(
        xy=center, width=semi_a*2, height=semi_b*2, angle=angle,
        facecolor="none", edgecolor=color, linestyle=ls, linewidth=lw
    ))

# =========================================================
# 描画（図幅を広くしてラベル切れを防ぐ）
# =========================================================
fig, ax = plt.subplots(figsize=(13, 9))
ax.axhline(0, color="#CCCCCC", lw=0.6)
ax.axvline(0, color="#CCCCCC", lw=0.6)

for cname, cfg in clusters.items():
    pts = mds_df.loc[cfg["cases"]][["Dim1","Dim2"]].values
    draw_ellipse(ax, pts, cfg["color"], cfg["ls"], cfg["lw"],
                 cfg["scale"], cfg["min_h"])

# マーカー
non_ainu = [c for c in cases if c not in {EDO_KEY, MEIJI_KEY}]
ax.scatter(mds_df.loc[non_ainu,"Dim1"], mds_df.loc[non_ainu,"Dim2"],
           s=55, facecolors="white", edgecolors="black", zorder=4)
for key in [EDO_KEY, MEIJI_KEY]:
    x, y = mds_df.loc[key, ["Dim1","Dim2"]]
    ax.scatter(x, y, s=90, marker="D", facecolors="white",
               edgecolors="black", linewidths=1.5, zorder=5)

# アイヌ間の変化線
x_e,y_e = mds_df.loc[EDO_KEY,   ["Dim1","Dim2"]]
x_m,y_m = mds_df.loc[MEIJI_KEY, ["Dim1","Dim2"]]
ax.plot([x_e,x_m],[y_e,y_m], color="black", lw=1.2,
        ls="--", alpha=0.6, zorder=3)

# =========================================================
# ラベル（リーダーライン付き）
# =========================================================
cx, cy = mds_df["Dim1"].mean(), mds_df["Dim2"].mean()
SCALE  = 7.0

for case in non_ainu:
    x,y   = mds_df.loc[case, ["Dim1","Dim2"]]
    label = labels_bilingual[case]
    dx,dy = x-cx, y-cy; norm = np.sqrt(dx**2+dy**2)+1e-6
    ox,oy = SCALE*dx/norm, SCALE*dy/norm
    ha    = "left"   if ox>=0 else "right"
    va    = "bottom" if oy>=0 else "top"
    ax.plot([x, x+ox*0.55],[y, y+oy*0.55],
            color="black", lw=0.5, alpha=0.5, zorder=3)
    ax.text(x+ox, y+oy, label,
            fontsize=8.5, ha=ha, va=va, zorder=5, linespacing=1.3)

# Ainu(Edo)ラベル: 右上へ
ax.plot([x_e, x_e+9],[y_e, y_e+7], color="black", lw=0.9, zorder=3)
ax.text(x_e+10, y_e+7, labels_bilingual[EDO_KEY],
        fontsize=10, fontweight="bold",
        ha="left", va="bottom", linespacing=1.3, zorder=6)

# Ainu(Meiji)ラベル: 右側（切れないよう右端に余裕）
ax.plot([x_m, x_m+4],[y_m, y_m-6], color="black", lw=0.9, zorder=3)
ax.text(x_m+5, y_m-6, labels_bilingual[MEIJI_KEY],
        fontsize=10, fontweight="bold",
        ha="left", va="top", linespacing=1.3, zorder=6)

# =========================================================
# 凡例（薄→濃の順）
# =========================================================
legend_elements = [
    Line2D([0],[0], ls=cfg["ls"], lw=cfg["lw"], color=cfg["color"],
           label=cfg["label"])
    for cfg in [clusters["入植同化型"], clusters["移行複合型"],
                clusters["東アジア帝国型"], clusters["搾取型"]]
]
ax.legend(handles=legend_elements, loc="lower left",
          fontsize=8.5, framealpha=0.93)

ax.set_xlabel("次元1 / Dimension 1（類似度空間）", fontsize=10)
ax.set_ylabel("次元2 / Dimension 2（類似度空間）", fontsize=10)
ax.set_title(
    "植民地統治プロファイルのMDS配置図（５因子空間・類似度基準）\n"
    "MDS Plot of Colonial Governance Profiles (5-Factor Similarity Space)",
    fontsize=11
)

# x軸右端を広げてラベル切れを防ぐ
xlim = ax.get_xlim(); ax.set_xlim(xlim[0], xlim[1]+8)

plt.tight_layout()
out = os.path.join(BASE_DIR, "mds_k4.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
