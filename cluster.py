# cluster.py — Case-level hierarchical clustering (5 factors)
#
# 【入力】 analysis_case_factor_means.csv  ← analyze.py が生成
# 【出力】 case_dendrogram_5factors.png
#
# 【実行例】
#   python3 cluster.py
#   python3 cluster.py --k 3
#   python3 cluster.py --input /path/to/means.csv

import os
import argparse
import warnings
import platform
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# パス設定
# ---------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "analysis_case_factor_means.csv")
OUT_IMG   = os.path.join(BASE_DIR, "case_dendrogram_5factors.png")

FACTORS = ["Land", "Labor", "Culture", "Political", "Economic"]

# ---------------------------------------------------------
# 日本語フォント（OS別フォールバック）
# ---------------------------------------------------------
def set_japanese_font() -> None:
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    candidates = {
        "Darwin":  ["Hiragino Sans", "Hiragino Maru Gothic Pro"],
        "Windows": ["Meiryo", "Yu Gothic", "MS Gothic"],
        "Linux":   ["Noto Sans CJK JP", "IPAexGothic", "VL Gothic"],
    }
    for font in candidates.get(platform.system(), []) + ["DejaVu Sans"]:
        if font in available:
            rcParams["font.family"] = font
            return
    rcParams["font.family"] = "DejaVu Sans"

set_japanese_font()
rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------
# ラベル・カラー定義
# ---------------------------------------------------------
LABEL_MAP = {
    "Ainu (Edo Period - Basho Ukeoi)":            "アイヌ（場所請負制期）\nAinu (Edo)",
    "Ainu (Meiji Period - Former Aborigine Law)":  "アイヌ（北海道入植植民期）\nAinu (Meiji)",
    "Aboriginal Australians":                      "豪州アボリジニ\nAboriginal Aus.",
    "Maori (New Zealand)":                         "マオリ\nMaori",
    "Native American (US)":                        "米国先住民\nNative American",
    "Taiwan (Japanese Rule)":                      "台湾（日本統治）\nTaiwan",
    "Korea (Japanese Rule)":                       "朝鮮（日本統治）\nKorea",
    "Indonesia (Dutch East Indies)":               "インドネシア（蘭領）\nIndonesia",
    "Bengal (British India)":                      "ベンガル（英領）\nBengal",
    "Ireland":                                     "アイルランド\nIreland",
    "Ryukyu (Okinawa)":                            "琉球（沖縄）\nRyukyu",
}

CLUSTER_COLORS = {
    1: "#C0C0C0",
    2: "#77787B",
    3: "#4C444D",
}
CLUSTER_LABELS = {
    1: "Cluster 1（入植型）",
    2: "Cluster 2（高土地型）",
    3: "Cluster 3（搾取型）",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ward 法階層クラスタリング + デンドログラム")
    parser.add_argument("--input", default=INPUT_CSV, metavar="FILE")
    parser.add_argument("--k",     type=int, default=4, metavar="N",
                        help="クラスター数（デフォルト: 3）")
    parser.add_argument("--out",   default=OUT_IMG, metavar="FILE")
    args = parser.parse_args()

    # --- 読み込み ---
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"入力CSVが見つかりません: {args.input}")

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    missing = ({"Case"} | set(FACTORS)) - set(df.columns)
    if missing:
        raise ValueError(f"必須列が不足: {missing}")

    case_means  = df.set_index("Case")[FACTORS]
    labels_disp = [LABEL_MAP.get(c, c) for c in case_means.index]

    # --- クラスタリング ---
    Z        = linkage(case_means.values, method="ward")
    clusters = fcluster(Z, t=args.k, criterion="maxclust")
    cut_h    = float(Z[-(args.k - 1), 2])

    # ---- コンソール出力 ----
    cluster_df = pd.DataFrame({"Case": case_means.index, "Cluster": clusters}).sort_values("Cluster")
    print(f"\n=== Ward クラスター割当（k={args.k}） ===")
    for k in sorted(cluster_df["Cluster"].unique()):
        members = cluster_df[cluster_df["Cluster"] == k]["Case"].tolist()
        print(f"\n  Cluster {k}:")
        for m in members:
            print(f"    {m}")

    cm = case_means.copy()
    cm["Cluster"] = clusters
    centroid = cm.groupby("Cluster")[FACTORS].mean().round(3)
    print(f"\n=== クラスター重心スコア ===")
    print(centroid.to_string())

    dist_sq = pd.DataFrame(
        squareform(pdist(case_means.values, metric="euclidean")),
        index=case_means.index, columns=case_means.index
    ).round(3)
    print(f"\n=== 事例間ユークリッド距離行列 ===")
    print(dist_sq.to_string())

    # ---- デンドログラム描画 ----
    # リンクカラー関数（同一クラスター内は対応色、異クラスター間は黒）
    case_list   = list(case_means.index)
    cluster_map = {case_list[i]: int(clusters[i]) for i in range(len(case_list))}

    def link_color_func(link_id: int) -> str:
        n = len(case_list)
        if link_id < n:
            return CLUSTER_COLORS.get(cluster_map[case_list[link_id]], "black")
        return "black"

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    dendrogram(
        Z,
        labels=labels_disp,
        orientation="right",
        link_color_func=link_color_func,
        ax=ax,
    )

    ax.axvline(x=cut_h, color="#444444", linestyle="--", linewidth=1.2)
    ax.text(cut_h + 0.002, ax.get_ylim()[0] + 2,
            f"k={args.k} カット距離={cut_h:.3f}",
            fontsize=8.5, color="#444444", va="bottom")

    # 凡例
    legend_handles = [
        Patch(facecolor=CLUSTER_COLORS.get(k, "gray"),
              label=CLUSTER_LABELS.get(k, f"Cluster {k}"))
        for k in range(1, args.k + 1)
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.9)

    ax.set_xlabel("距離（Ward 法）", fontsize=11)
    ax.set_title(
        f"植民地統治類型の階層クラスタリング（Ward 法, k={args.k}）\n"
        "Hierarchical Clustering of Colonial Governance Profiles (5-Factor Space)",
        fontsize=12, pad=12,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ デンドログラム保存: {args.out}")


if __name__ == "__main__":
    main()