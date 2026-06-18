# chart.py — 5-factor radar charts
#
# 【入力】 analysis_case_factor_means.csv  ← analyze.py が生成
# 【出力】
#   ainu_radar_5factors.png         — Ainu Edo vs Meiji 比較
#   all_cases_radar_5factors.png    — 全事例レーダーチャート（グリッド表示）
#
# 【実行例】
#   python3 chart.py
#   python3 chart.py --no-all       # 全事例チャートをスキップ
#   python3 chart.py --max-score 5  # 軸の最大値を指定

import os
import argparse
import platform
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# パス設定
# ---------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "analysis_case_factor_means.csv")
OUT_AINU  = os.path.join(BASE_DIR, "ainu_radar_5factors.png")
OUT_ALL   = os.path.join(BASE_DIR, "all_cases_radar_5factors.png")

FACTORS    = ["Land", "Labor", "Culture", "Political", "Economic"]
FACTORS_JP = ["土地収奪", "労働搾取", "文化同化", "政治統制", "経済収奪"]

AINU_EDO   = "Ainu (Edo Period - Basho Ukeoi)"
AINU_MEIJI = "Ainu (Meiji Period - Former Aborigine Law)"

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
# 事例別カラー定義
# ---------------------------------------------------------
CASE_COLORS = {
    AINU_EDO:                              "#C0C0C0",
    AINU_MEIJI:                            "#4C444D",
    "Maori (New Zealand)":                 "#4C444D",
    "Native American (US)":                "#C0C0C0",
    "Aboriginal Australians":              "#4C444D",
    "Taiwan (Japanese Rule)":              "#77787B",
    "Korea (Japanese Rule)":               "#77787B",
    "Indonesia (Dutch East Indies)":       "#333333",
    "Bengal (British India)":              "#333333",
    "Ireland":                             "#C0C0C0",
    "Ryukyu (Okinawa)":                    "#77787B",
}

CASE_LABELS_JP = {
    AINU_EDO:                              "アイヌ（場所請負制期）",
    AINU_MEIJI:                            "アイヌ（北海道入植植民期）",
    "Maori (New Zealand)":                 "マオリ",
    "Native American (US)":               "米国先住民",
    "Aboriginal Australians":             "豪州アボリジニ",
    "Taiwan (Japanese Rule)":             "台湾（日本統治）",
    "Korea (Japanese Rule)":              "朝鮮（日本統治）",
    "Indonesia (Dutch East Indies)":      "インドネシア（蘭領）",
    "Bengal (British India)":             "ベンガル（英領）",
    "Ireland":                            "アイルランド",
    "Ryukyu (Okinawa)":                   "琉球（沖縄）",
}


# ---------------------------------------------------------
# レーダーチャート描画ヘルパー
# ---------------------------------------------------------
def make_radar_axes(fig, rect, n_factors: int):
    """極座標サブプロットを作成して返す。"""
    ax = fig.add_axes(rect, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    return ax


def draw_radar(
    ax,
    angles: list,
    values: list,
    label: str,
    color: str,
    linestyle: str = "-",
    linewidth: float = 2.0,
    fill_alpha: float = 0.10,
) -> None:
    """1事例分のレーダー折れ線と塗りつぶしを描く。"""
    ax.plot(angles, values, linewidth=linewidth, linestyle=linestyle,
            color=color, label=label)
    ax.fill(angles, values, alpha=fill_alpha, color=color)


def setup_radar_ax(ax, angles: list, labels: list, max_val: float, n_ticks: int = 5) -> None:
    """軸ラベル・グリッドを設定する。"""
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    tick_vals = np.linspace(0, max_val, n_ticks + 1)[1:]
    ax.set_yticks(tick_vals)
    ax.set_yticklabels([f"{v:.2f}" for v in tick_vals], fontsize=7, color="#666666")
    ax.set_ylim(0, max_val)
    ax.grid(True, linestyle="--", alpha=0.4)


# ---------------------------------------------------------
# Chart A: Ainu Edo vs Meiji 比較
# ---------------------------------------------------------
def plot_ainu_comparison(means: pd.DataFrame, max_val: float, out_path: str) -> None:
    for case in [AINU_EDO, AINU_MEIJI]:
        if case not in means.index:
            print(f"  ⚠️  '{case}' が CSV に存在しません。スキップします。")
            return

    n       = len(FACTORS)
    angles  = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist() + [0]

    edo_vals   = means.loc[AINU_EDO,   FACTORS].tolist() + [means.loc[AINU_EDO,   FACTORS[0]]]
    meiji_vals = means.loc[AINU_MEIJI, FACTORS].tolist() + [means.loc[AINU_MEIJI, FACTORS[0]]]

    fig = plt.figure(figsize=(7, 7), facecolor="#FAFAFA")
    ax  = make_radar_axes(fig, [0.1, 0.1, 0.8, 0.8], n)
    setup_radar_ax(ax, angles, FACTORS_JP, max_val)

    draw_radar(ax, angles, edo_vals,
               label=CASE_LABELS_JP.get(AINU_EDO, AINU_EDO),
               color=CASE_COLORS[AINU_EDO], linestyle="-")
    draw_radar(ax, angles, meiji_vals,
               label=CASE_LABELS_JP.get(AINU_MEIJI, AINU_MEIJI),
               color=CASE_COLORS[AINU_MEIJI], linestyle="--")

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              frameon=False, fontsize=10)
    ax.set_title(
        "アイヌ植民地統治の類型転換\nAinu Colonial Governance Transformation",
        y=1.12, fontsize=12,
    )

    # 差分注釈
    diff = means.loc[AINU_MEIJI, FACTORS] - means.loc[AINU_EDO, FACTORS]
    most_inc = diff.idxmax()
    most_dec = diff.idxmin()
    note = (f"最大増加: {most_inc} ({diff[most_inc]:+.3f})\n"
            f"最大減少: {most_dec} ({diff[most_dec]:+.3f})")
    fig.text(0.02, 0.02, note, fontsize=8.5, color="#444444",
             va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Ainu レーダーチャート保存: {out_path}")


# ---------------------------------------------------------
# Chart B: 全事例グリッドレーダーチャート
# ---------------------------------------------------------
def plot_all_cases_grid(means: pd.DataFrame, max_val: float, out_path: str) -> None:
    cases = list(means.index)
    n_cases = len(cases)
    n_cols  = 4
    n_rows  = (n_cases + n_cols - 1) // n_cols
    n       = len(FACTORS)
    angles  = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist() + [0]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.5, n_rows * 3.5),
        subplot_kw=dict(polar=True),
        facecolor="#FAFAFA",
    )
    fig.suptitle(
        "植民地統治 5因子プロファイル（全事例）\nColonial Governance 5-Factor Profiles",
        fontsize=13, y=1.01,
    )
    axes_flat = axes.flatten()

    for idx, case in enumerate(cases):
        ax = axes_flat[idx]
        ax.set_facecolor("#FAFAFA")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        setup_radar_ax(ax, angles, FACTORS_JP, max_val, n_ticks=3)

        vals  = means.loc[case, FACTORS].tolist() + [means.loc[case, FACTORS[0]]]
        color = CASE_COLORS.get(case, "#888888")
        draw_radar(ax, angles, vals, label="", color=color,
                   linewidth=1.8, fill_alpha=0.18)

        # タイトル
        title = CASE_LABELS_JP.get(case, case)
        ax.set_title(title, fontsize=8.5, pad=10,
                     color="#222222", fontweight="bold")

        # スコアを軸上に表示
        for i, (angle, val, factor) in enumerate(zip(angles[:-1], vals[:-1], FACTORS)):
            ax.annotate(
                f"{val:.2f}",
                xy=(angle, val),
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=6.5,
                ha="center",
                color=color,
            )

    # 余白のサブプロットを非表示
    for idx in range(n_cases, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 全事例レーダーチャート保存: {out_path}")


# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="レーダーチャート生成")
    parser.add_argument("--input",     default=INPUT_CSV, metavar="FILE")
    parser.add_argument("--out-ainu",  default=OUT_AINU,  metavar="FILE")
    parser.add_argument("--out-all",   default=OUT_ALL,   metavar="FILE")
    parser.add_argument("--no-all",    action="store_true",
                        help="全事例グリッドチャートをスキップ")
    parser.add_argument("--max-score", type=float, default=None, metavar="FLOAT",
                        help="レーダー軸の最大値（省略時はデータの最大値 × 1.15）")
    args = parser.parse_args()

    # --- 読み込み ---
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"入力CSVが見つかりません: {args.input}")

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    missing = ({"Case"} | set(FACTORS)) - set(df.columns)
    if missing:
        raise ValueError(f"必須列が不足: {missing}")

    means = df.set_index("Case")[FACTORS].astype(float)

    # 軸最大値の決定
    max_val = args.max_score if args.max_score else float(means.max().max()) * 1.15

    print(f"  入力: {args.input}  ({len(means)} 事例)")
    print(f"  スコアレンジ: 0.0 – {max_val:.3f}\n")

    # Chart A: Ainu 比較
    plot_ainu_comparison(means, max_val, args.out_ainu)

    # Chart B: 全事例グリッド
    if not args.no_all:
        plot_all_cases_grid(means, max_val, args.out_all)


if __name__ == "__main__":
    main()