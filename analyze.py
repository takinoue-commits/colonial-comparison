"""
analyze.py — Phase 5 Statistical Analysis

INPUT:
    main_analysis_results.csv

OUTPUT (same directory as this script):
    analysis_case_factor_means.csv   ← 事例×5因子の平均スコア表
    analysis_case_factor_stats.csv   ← 事例×5因子の基本統計（mean/median/SD/n/min/max）
    ainu_statistical_tests.csv       ← Ainu Edo vs Meiji の Mann-Whitney U + Cliff's Delta
    ainu_diff_vector.csv             ← Ainu 両期間の差分ベクトル（Edo mean / Meiji mean / 差分）
    analysis_pairwise_distances.csv  ← 事例間ユークリッド距離行列
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------
# パス設定
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_CSV  = os.path.join(BASE_DIR, "main_analysis_results.csv")

OUT_MEANS     = os.path.join(BASE_DIR, "analysis_case_factor_means.csv")
OUT_STATS     = os.path.join(BASE_DIR, "analysis_case_factor_stats.csv")
OUT_AINU_TEST = os.path.join(BASE_DIR, "ainu_statistical_tests.csv")   # 既存ファイル名に合わせる
OUT_DIFF_VEC  = os.path.join(BASE_DIR, "ainu_diff_vector.csv")          # 既存ファイル名に合わせる
OUT_DIST      = os.path.join(BASE_DIR, "analysis_pairwise_distances.csv")

# ---------------------------------------------------------
# 定数
# ---------------------------------------------------------
FACTORS = ["Land", "Labor", "Culture", "Political", "Economic"]

AINU_EDO   = "Ainu (Edo Period - Basho Ukeoi)"
AINU_MEIJI = "Ainu (Meiji Period - Former Aborigine Law)"


# ---------------------------------------------------------
# Cliff's Delta + 効果量ラベル
# ---------------------------------------------------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's Delta（効果量）を返す。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    greater = np.sum(x[:, None] > y[None, :])
    less    = np.sum(x[:, None] < y[None, :])
    return float((greater - less) / (len(x) * len(y)))


def delta_label(d: float) -> str:
    """Cliff's Delta の効果量を解釈する（Romano et al. 2006 基準）。"""
    a = abs(d)
    if a < 0.147:
        return "negligible"
    elif a < 0.330:
        return "small"
    elif a < 0.474:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------
# 出力ファイルの存在確認ヘルパー
# ---------------------------------------------------------
def assert_written(path: str) -> None:
    if not os.path.isfile(path):
        raise RuntimeError(f"❌ ファイルが生成されませんでした: {path}")


# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("  analyze.py — Phase 5 統計分析")
    print(f"  スクリプトディレクトリ: {BASE_DIR}")
    print("=" * 72)

    # ---------- 入力チェック ----------
    if not os.path.isfile(INPUT_CSV):
        print(f"  ❌ 入力ファイルが見つかりません: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"  入力: {len(df)} 行  事例数: {df['Case'].nunique()}")

    required = {"Case"} | set(FACTORS)
    missing  = required - set(df.columns)
    if missing:
        print(f"  ❌ 必須列が不足しています: {missing}")
        sys.exit(1)

    # ==========================================
    # 1. 事例×5因子 平均スコア（Table 2 用）
    # ==========================================
    case_means = (
        df.groupby("Case")[FACTORS]
        .mean()
        .round(3)
        .reset_index()
    )
    case_means.to_csv(OUT_MEANS, index=False, encoding="utf-8-sig")
    assert_written(OUT_MEANS)

    print("\n  ■ 事例別平均スコア")
    print(case_means.to_string(index=False))

    # ==========================================
    # 2. 事例×5因子 基本統計（mean / median / SD / n / min / max）
    # ==========================================
    stat_rows = []
    for case, sub in df.groupby("Case"):
        for f in FACTORS:
            vals = sub[f].dropna()
            stat_rows.append({
                "Case":   case,
                "Factor": f,
                "n":      len(vals),
                "Mean":   round(vals.mean(),   3),
                "SD":     round(vals.std(ddof=1), 3) if len(vals) > 1 else 0.0,
                "Median": round(vals.median(), 3),
                "Min":    round(vals.min(),    3),
                "Max":    round(vals.max(),    3),
            })

    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(OUT_STATS, index=False, encoding="utf-8-sig")
    assert_written(OUT_STATS)

    # ==========================================
    # 3. Ainu Edo vs Meiji — Mann-Whitney U + Cliff's Delta
    # ==========================================
    test_rows = []
    print("\n  ■ Ainu Edo vs Meiji — Mann-Whitney U / Cliff's Delta")
    print(f"  {'Dimension':<12} {'U':>8} {'p':>8} {'δ':>8} {'Effect':>12}")
    print("  " + "-" * 52)

    for f in FACTORS:
        x = df[df["Case"] == AINU_EDO][f].dropna().to_numpy()
        y = df[df["Case"] == AINU_MEIJI][f].dropna().to_numpy()

        if len(x) == 0 or len(y) == 0:
            print(f"  ⚠️  {f}: データなし（スキップ）")
            continue

        u_stat, p_val = mannwhitneyu(x, y, alternative="two-sided",
                                     method="asymptotic")
        delta = cliffs_delta(x, y)
        label = delta_label(delta)
        sig   = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")

        test_rows.append({
            "Dimension":    f,          # 既存 ainu_statistical_tests.csv の列名に合わせる
            "U_statistic":  round(u_stat, 2),
            "p_value":      round(p_val,  5),
            "Cliffs_Delta": round(delta,  3),
            "Effect_Size":  label,
            "Sig":          sig,
            "n_Edo":        len(x),
            "n_Meiji":      len(y),
        })

        print(f"  {f:<12} {u_stat:>8.1f} {p_val:>8.4f} {delta:>8.3f} {label:>12} {sig}")

    ainu_tests = pd.DataFrame(test_rows)
    ainu_tests.to_csv(OUT_AINU_TEST, index=False, encoding="utf-8-sig")
    assert_written(OUT_AINU_TEST)

    # ==========================================
    # 4. Ainu 差分ベクトル（Meiji − Edo）
    # ==========================================
    diff_rows = []
    print("\n  ■ Ainu 差分ベクトル（Meiji − Edo）")
    print(f"  {'Dimension':<30} {'Edo_Mean':>10} {'Meiji_Mean':>12} {'Diff (M-E)':>12}")
    print("  " + "-" * 66)

    for f in FACTORS:
        edo_vals   = df[df["Case"] == AINU_EDO][f].dropna()
        meiji_vals = df[df["Case"] == AINU_MEIJI][f].dropna()

        edo_mean   = round(edo_vals.mean(),   3) if len(edo_vals)   > 0 else None
        meiji_mean = round(meiji_vals.mean(), 3) if len(meiji_vals) > 0 else None
        diff       = round(meiji_mean - edo_mean, 3) if (
            edo_mean is not None and meiji_mean is not None) else None

        diff_rows.append({
            "Dimension":           f,          # 既存 ainu_diff_vector.csv の列名
            "Edo_Mean":            edo_mean,
            "Meiji_Mean":          meiji_mean,
            "Difference_(Meiji_minus_Edo)": diff,
        })

        sign = ("+" if diff > 0 else "") if diff is not None else "N/A"
        print(f"  {f:<30} {str(edo_mean):>10} {str(meiji_mean):>12} "
              f"{sign + str(diff) if diff is not None else 'N/A':>12}")

    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(OUT_DIFF_VEC, index=False, encoding="utf-8-sig")
    assert_written(OUT_DIFF_VEC)

    # ==========================================
    # 5. 事例間ユークリッド距離行列
    # ==========================================
    pivot = (
        df.groupby("Case")[FACTORS]
        .mean()
    )
    dist_matrix = pd.DataFrame(
        squareform(pdist(pivot.values, metric="euclidean")),
        index=pivot.index,
        columns=pivot.index,
    ).round(3)

    dist_matrix.to_csv(OUT_DIST, encoding="utf-8-sig")
    assert_written(OUT_DIST)

    # Ainu 近接ランキング（コンソール表示）
    print("\n  ■ Ainu 各期間との距離ランキング（近い順）")
    for ainu_case in [AINU_EDO, AINU_MEIJI]:
        if ainu_case not in dist_matrix.index:
            continue
        row = dist_matrix[ainu_case].drop(index=ainu_case).sort_values()
        label = ainu_case.split("(")[1].rstrip(")")
        print(f"\n  [{label}]")
        for case, d in row.items():
            print(f"    {d:.3f}  {case}")

    # ==========================================
    # 完了
    # ==========================================
    print("\n" + "=" * 72)
    print("  ✅ 全出力ファイルの生成を確認しました")
    print(f"    {OUT_MEANS}")
    print(f"    {OUT_STATS}")
    print(f"    {OUT_AINU_TEST}")
    print(f"    {OUT_DIFF_VEC}")
    print(f"    {OUT_DIST}")
    print("=" * 72)


if __name__ == "__main__":
    main()