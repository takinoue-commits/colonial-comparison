from typing import DefaultDict, Dict, List, Optional, Set, Tuple
# =========================================================
# prescreen.py  Phase 4: 最終論文選抜
# =========================================================

import os
import re
import sys
import argparse
from collections import defaultdict

import pandas as pd

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
KW_CSV        = os.path.join(BASE_DIR, "screening_report_anchor_kw.csv")
QWEN_CSV      = os.path.join(BASE_DIR, "screening_report_qwen.csv")
SELECTED_CSV  = os.path.join(BASE_DIR, "selected_papers.csv")
CACHE_DIR     = os.path.join(BASE_DIR, "extraction_cache")

KW_THRESHOLD       = 0.15
KW_BOOST_THRESHOLD = 0.10
PAPERS_PER_CASE    = 10
MAX_CASES_PER_FILE = 4      # 1論文が割り当てられる最大事例数
MIN_WORD_COUNT     = 2000   # これ未満の論文はテキスト抽出不完全とみなし除外

# 事例ごとの KW 閾値オーバーライド（省略した事例は KW_THRESHOLD を使用）
PER_CASE_THRESHOLD: Dict[str, float] = {
    "Taiwan (Japanese Rule)": 0.05,   # 候補不足のため閾値をさらに緩和
}

# 事例ごとの MIN_WORD_COUNT オーバーライド（省略した事例は MIN_WORD_COUNT を使用）
PER_CASE_MIN_WORDS: Dict[str, int] = {
    "Taiwan (Japanese Rule)": 1500,   # 候補不足のため語数下限を緩和
}

PREDEFINED_CASES = [
    "Ainu (Edo Period - Basho Ukeoi)",
    "Ainu (Meiji Period - Former Aborigine Law)",
    "Taiwan (Japanese Rule)",       # 候補不足のため先行処理（3番目に移動）
    "Maori (New Zealand)",
    "Native American (US)",
    "Aboriginal Australians",
    "Korea (Japanese Rule)",
    "Indonesia (Dutch East Indies)",
    "Bengal (British India)",
    "Ireland",
    "Ryukyu (Okinawa)",
]

# アイヌ事例専用論文セット
# ここに列挙したファイルは Ainu Edo / Ainu Meiji 以外への割当を
# EXCLUDE_BY_CASE に関係なく強制的に禁止する
AINU_ONLY_FILES: Set[str] = {
    "Hirano-SettlerColonialismEcologyExpropriation-2023.pdf",
}

AINU_CASES: Set[str] = {
    "Ainu (Edo Period - Basho Ukeoi)",
    "Ainu (Meiji Period - Former Aborigine Law)",
}

# 強制包含リスト: {ファイル名: {事例名, ...}}
# キャッシュ欠落や wc=0 で MIN_WORD_COUNT に弾かれる論文を事例に強制追加する
# KW スコアが閾値以上であることは引き続き確認する
FORCE_INCLUDE: Dict[str, Set[str]] = {
    "Elite_Formation_and_Transformation_in_Co.pdf": {"Taiwan (Japanese Rule)"},
    "Tsai-DiariesEverydayLife-2013.pdf":            {"Taiwan (Japanese Rule)"},
}

EXCLUDE_BY_CASE: Dict[str, Set[str]] = {

    # ── ファイル名ラベルと実内容が逆転 ──────────────────────────
    "Mason2012_Meiji.pdf":     {"Ainu (Edo Period - Basho Ukeoi)"},
    "Howell2004_Edo.pdf":      {"Ainu (Meiji Period - Former Aborigine Law)"},
    "WatanabeIto2024_Edo.pdf": {"Ainu (Meiji Period - Former Aborigine Law)"},
    # 琉球・北海道比較論文 → アイヌ事例ではなく琉球事例として扱う
    "On-the-Peripheries-of-the-Japanese-Archipelago-Ryukyu-and-Hokkaido_2023_Cambridge-University-Press.pdf": {
        "Ainu (Edo Period - Basho Ukeoi)",
        "Ainu (Meiji Period - Former Aborigine Law)",
    },

    # ── 明らかな誤割当 ────────────────────────────────────────
    "Wickstrum2015_Korea.pdf":                    {"Taiwan (Japanese Rule)"},
    "Bengal_Regulation_10_of_1804_and_Martial.pdf": {"Ireland"},
    "Karafuto_as_a_Border_Island_of_the_Empir.pdf": {"Ryukyu (Okinawa)"},
    "okada2012.pdf":                              {"Native American (US)"},
    "Decolonisation-and-the-Pacific-Indigenous-Globalisation-and-the-Ends-of-Empire_2016_Cambridge-University-Press.pdf":
        {"Indonesia (Dutch East Indies)"},
    "Chatterjee-ColonialStatePeasant-1986.pdf":   {"Ireland"},
    "The-Plantations-Outsides-The-Work-of-Settlement-in-Kalimpong-India_2021_Cambridge-University-Press.pdf":
        set(PREDEFINED_CASES),
    "Youth_Baseball_and_Colonial_Identity_in.pdf": {"Taiwan (Japanese Rule)"},
    "Beyond-Fistfights-and-Basketball-Reclaiming-Native-American-Masculinity_2024_Multidisciplinary-Digital-Publishing-Institute-MDPI.pdf":
        {"Native American (US)"},
    # 日本の同化政策論文 → Ireland への割当は無関係
    "Japanese_Assimilation_Policies_in_Coloni.pdf": {"Ireland"},
    # 台湾日記研究 → Ainu Meiji への割当は無関係
    "Tsai-DiariesEverydayLife-2013.pdf": {
        "Ainu (Edo Period - Basho Ukeoi)",
        "Ainu (Meiji Period - Former Aborigine Law)",
        "Native American (US)", "Aboriginal Australians",
        "Korea (Japanese Rule)", "Indonesia (Dutch East Indies)",
        "Bengal (British India)", "Ireland", "Ryukyu (Okinawa)",
    },
    # スポーツ論文 → 全事例から除外（Native American除外済みだが Indonesia に流入）
    "Beyond-Fistfights-and-Basketball-Reclaiming-Native-American-Masculinity_2024_Multidisciplinary-Digital-Publishing-Institute-MDPI.pdf":
        set(PREDEFINED_CASES),
    # 大英帝国古典論文 → Dutch East Indies との関連なし
    "Classics_and_Imperialism_in_the_British.pdf": {"Indonesia (Dutch East Indies)"},
    # アメリカ先住民研究論文（ファイル名に _America）→ Indonesia への割当は無関係
    "Panich2013_America.pdf": {"Indonesia (Dutch East Indies)"},
    # 植民地建築汎用論文 → Dutch East Indies 固有でない
    "Theorizing_colonial_architecture_and_urb.pdf": {"Indonesia (Dutch East Indies)"},
    # NZラグビー論文 → 植民地統治と無関係
    "Postcolonial-anxieties-and-the-browning-of-New-Zealand-rugby_2012_.pdf":
        set(PREDEFINED_CASES),
    # 台湾論文 → Korea への割当は無関係
    "Chan2020_Taiwan.pdf": {"Korea (Japanese Rule)"},

    # ── 同一語数重複ペア ─────────────────────────────────────
    "article-463_Okinawa.pdf": {"Ryukyu (Okinawa)"},
    "Mays-2007.pdf":           {"Aboriginal Australians"},

    # ── ファイル名不明・内容検証不可 ─────────────────────────
    "2.pdf":              set(PREDEFINED_CASES),
    "7.pdf":              set(PREDEFINED_CASES),
    "read australia.pdf": set(PREDEFINED_CASES),
    # レビュー論文として除外
    "Recovering_the_Subject_in_the_Shadows_of.pdf": set(PREDEFINED_CASES),
    "0822__.pdf":         set(PREDEFINED_CASES),

    # ── 事例割当の誤り ───────────────────────────────────────
    "Peng-SocioeconomicStatusAinu-1974.pdf": {
        "Maori (New Zealand)", "Native American (US)", "Aboriginal Australians",
        "Taiwan (Japanese Rule)", "Korea (Japanese Rule)",
        "Indonesia (Dutch East Indies)", "Bengal (British India)",
        "Ireland", "Ryukyu (Okinawa)",
    },

    # ── アイヌ専用指定 ──────────────────────────────────────
    # Hirano 2023 はアイヌ植民地主義の専論 → アイヌ以外の全事例から除外
    "Hirano-SettlerColonialismEcologyExpropriation-2023.pdf": {
        "Maori (New Zealand)", "Native American (US)", "Aboriginal Australians",
        "Taiwan (Japanese Rule)", "Korea (Japanese Rule)",
        "Indonesia (Dutch East Indies)", "Bengal (British India)",
        "Ireland", "Ryukyu (Okinawa)",
    },
}

def _build_exclude() -> None:
    EXCLUDE_BY_CASE["Rahman_Clarke_Byrne_Full.pdf"] = set(PREDEFINED_CASES)

_build_exclude()


def load_qwen_results(qwen_csv: str) -> Dict[str, Set[str]]:
    if not os.path.exists(qwen_csv):
        return {}
    result: Dict[str, Set[str]] = defaultdict(set)
    try:
        df = pd.read_csv(qwen_csv, encoding="utf-8-sig")
        for _, row in df.iterrows():
            fn = str(row.get("Filename", "")).strip()
            if not fn:
                continue
            for col in ("qwen_case_1", "qwen_case_2"):
                val = str(row.get(col, "")).strip()
                if val and val not in ("None", "", "nan"):
                    result[fn].add(val)
    except Exception as e:
        print(f"  [警告] Qwen CSV の読み込みに失敗しました: {e}")
        return {}
    return dict(result)


def _count_words(text: str) -> int:
    en = len(re.findall(r"[a-zA-Z]{2,}", text))
    ja = len(re.findall(r"[\u3040-\u9FFF]{2,}", text))
    return en + ja


def load_word_counts(cache_dir: str, filenames: List[str]) -> Dict[str, int]:
    wc_map: Dict[str, int] = {}
    for fn in filenames:
        txt_path = os.path.join(cache_dir, fn + ".txt")
        if os.path.exists(txt_path):
            try:
                with open(txt_path, encoding="utf-8") as f:
                    wc_map[fn] = _count_words(f.read())
            except Exception:
                wc_map[fn] = 0
        else:
            wc_map[fn] = 0
    return wc_map


def select_papers(
    df: pd.DataFrame,
    qwen: Dict[str, Set[str]],
    kw_threshold: float,
    boost_threshold: float,
    use_qwen: bool,
    wc_map: Dict[str, int],
) -> List[dict]:

    # ── 事前計算: 各ファイルが何事例で採用条件を満たすかをカウント ──────────
    # これにより「単一事例一致論文（is_single=True）」を事前に特定できる。
    # 単一事例一致論文はKWスコアが低くても優先して採用する。
    fn_case_count: DefaultDict[str, int] = defaultdict(int)
    for case in PREDEFINED_CASES:
        score_col = f"kw_score_{case}"
        if score_col not in df.columns:
            continue
        for _, row in df.iterrows():
            fn       = str(row["Filename"]).strip()
            kw_score = float(row.get(score_col, 0.0) or 0.0)
            wc_csv   = int(row.get("word_count", 0) or 0)
            wc       = wc_map.get(fn, wc_csv) if wc_csv == 0 else wc_csv

            if wc < PER_CASE_MIN_WORDS.get(case, MIN_WORD_COUNT):
                continue
            if fn in EXCLUDE_BY_CASE and case in EXCLUDE_BY_CASE[fn]:
                continue
            # アイヌ専用ファイルはアイヌ事例以外に割当不可（事前計算でも除外）
            if fn in AINU_ONLY_FILES and case not in AINU_CASES:
                continue

            ok_by_kw   = kw_score >= PER_CASE_THRESHOLD.get(case, kw_threshold)
            ok_by_qwen = (
                use_qwen
                and kw_score >= boost_threshold
                and fn in qwen
                and case in qwen[fn]
            )
            if ok_by_kw or ok_by_qwen:
                fn_case_count[fn] += 1

    # fn_case_count[fn] == 1 の論文が「単一事例一致論文」
    single_case_files: Set[str] = {
        fn for fn, cnt in fn_case_count.items() if cnt == 1
    }

    # ── 選抜ループ ──────────────────────────────────────────────────────────
    selected_rows: List[dict] = []
    use_count: DefaultDict[str, int] = defaultdict(int)

    for case in PREDEFINED_CASES:
        score_col = f"kw_score_{case}"
        if score_col not in df.columns:
            print(f"  ⚠️  [{case}]: 列が見つかりません ({score_col})")
            print("      recompute_kw_screen.py を先に実行してください。")
            continue

        candidates: List[dict] = []
        for _, row in df.iterrows():
            fn       = str(row["Filename"]).strip()
            kw_score = float(row.get(score_col, 0.0) or 0.0)
            gov_dens = float(row.get("governance_density", 0.0) or 0.0)
            wc_csv   = int(row.get("word_count", 0) or 0)
            wc       = wc_map.get(fn, wc_csv) if wc_csv == 0 else wc_csv

            force = fn in FORCE_INCLUDE and case in FORCE_INCLUDE[fn]
            if not force and wc < PER_CASE_MIN_WORDS.get(case, MIN_WORD_COUNT):
                continue
            if fn in EXCLUDE_BY_CASE and case in EXCLUDE_BY_CASE[fn]:
                continue
            # アイヌ専用ファイルはアイヌ事例以外に割当不可（選抜でも除外）
            if fn in AINU_ONLY_FILES and case not in AINU_CASES:
                continue

            ok_by_kw   = force or kw_score >= PER_CASE_THRESHOLD.get(case, kw_threshold)
            ok_by_qwen = (
                use_qwen
                and kw_score >= boost_threshold
                and fn in qwen
                and case in qwen[fn]
            )
            if not (ok_by_kw or ok_by_qwen):
                continue
            if use_count[fn] >= MAX_CASES_PER_FILE:
                continue

            is_single = fn in single_case_files

            candidates.append({
                "Filename":           fn,
                "Case_Override":      case,
                "kw_score":           round(kw_score, 3),
                "governance_density": round(gov_dens, 3),
                "word_count":         wc,
                "qwen_confirmed":     ok_by_qwen and not ok_by_kw,
                "is_single":          is_single,
            })

        if not candidates:
            print(f"  ⚠️  [{case}]: 0本（閾値を満たす論文なし）")
            continue

        # ソート優先順位:
        #   1. 単一事例一致論文を最優先（is_single=True が先）
        #   2. 同じグループ内では kw_score 降順
        #   3. 同スコアなら word_count 降順
        candidates.sort(
            key=lambda r: (
                0 if r["is_single"] else 1,   # 単一事例が先
                -r["kw_score"],
                -r["word_count"],
            )
        )

        top = candidates[:PAPERS_PER_CASE]

        n_single = sum(1 for r in top if r["is_single"])
        n_multi  = len(top) - n_single
        if len(top) < PAPERS_PER_CASE:
            print(f"  ⚠️  [{case}]: {len(top)}本（目標 {PAPERS_PER_CASE}本に未達）"
                  f"  単一:{n_single}  多事例:{n_multi}")
        else:
            print(f"  ✓  [{case[:42]}]  単一:{n_single}  多事例:{n_multi}")

        for r in top:
            use_count[r["Filename"]] += 1
            selected_rows.append(r)

    fn_count: DefaultDict[str, int] = defaultdict(int)
    for r in selected_rows:
        fn_count[r["Filename"]] += 1
    for r in selected_rows:
        r["Is_Multi_Case"] = fn_count[r["Filename"]] > 1

    return selected_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 最終論文選抜")
    parser.add_argument("--threshold",       type=float, default=KW_THRESHOLD,       metavar="FLOAT")
    parser.add_argument("--boost-threshold", type=float, default=KW_BOOST_THRESHOLD, metavar="FLOAT")
    parser.add_argument("--no-qwen",         action="store_true")
    parser.add_argument("--kw-csv",          default=KW_CSV,       metavar="FILE")
    parser.add_argument("--qwen-csv",        default=QWEN_CSV,     metavar="FILE")
    parser.add_argument("--out",             default=SELECTED_CSV, metavar="FILE")
    args = parser.parse_args()

    print("=" * 72)
    print("  prescreen.py — Phase 4 最終論文選抜")
    print(f"  KW閾値={args.threshold}  Qwen補強閾値={args.boost_threshold}"
          f"  Qwen使用={'なし' if args.no_qwen else 'あり'}")
    print("=" * 72)

    if not os.path.exists(args.kw_csv):
        print(f"  ❌ KWスコアCSVが見つかりません: {args.kw_csv}")
        sys.exit(1)

    df = pd.read_csv(args.kw_csv, encoding="utf-8-sig")
    if "Filename" not in df.columns:
        print("  ❌ CSVに Filename 列がありません。")
        sys.exit(1)
    print(f"  KW CSV: {len(df)} 件読み込み")

    use_qwen = not args.no_qwen
    qwen: Dict[str, Set[str]] = {}
    if use_qwen:
        qwen = load_qwen_results(args.qwen_csv)
        if qwen:
            print(f"  Qwen CSV: {len(qwen)} 件読み込み")
        else:
            print("  Qwen CSV: 未使用（ファイルなし、または --no-qwen 指定）")
            use_qwen = False

    print()

    wc_col = df.get("word_count", pd.Series([0] * len(df))).fillna(0).astype(int)
    wc_map: Dict[str, int] = {}
    if (wc_col == 0).all():
        print("  word_count=0 を検出 → extraction_cache からword_countを再計算中...")
        wc_map = load_word_counts(CACHE_DIR, df["Filename"].tolist())
        n_ok = sum(1 for v in wc_map.values() if v > 0)
        print(f"  word_count 算出: {n_ok}/{len(wc_map)} 件\n")

    selected = select_papers(
        df, qwen,
        kw_threshold=args.threshold,
        boost_threshold=args.boost_threshold,
        use_qwen=use_qwen,
        wc_map=wc_map,
    )

    if not selected:
        print("\n  ❌ 選抜結果が空です（全事例で 0 本）")
        sys.exit(1)

    out_cols = [
        "Filename", "Case_Override", "Is_Multi_Case",
        "kw_score", "word_count", "governance_density", "qwen_confirmed",
    ]
    df_sel = pd.DataFrame(selected)[out_cols]
    df_sel.to_csv(args.out, index=False, encoding="utf-8-sig")

    print()
    total_unique = len(df_sel["Filename"].unique())
    total_pairs  = len(df_sel)
    n_qwen_only  = int(df_sel["qwen_confirmed"].sum())

    for case in PREDEFINED_CASES:
        sub = df_sel[df_sel["Case_Override"] == case]
        if sub.empty:
            continue
        n_multi   = int(sub["Is_Multi_Case"].sum())
        n_single  = len(sub) - n_multi
        n_boosted = int(sub["qwen_confirmed"].sum())
        status    = "✓" if len(sub) >= PAPERS_PER_CASE else f"⚠️ {len(sub)}本"
        print(f"\n  ■ {case} ({status})"
              f"  単一:{n_single}  多事例:{n_multi}"
              + (f"  Qwen補強:{n_boosted}" if n_boosted else ""))
        for _, r in sub.sort_values(
            ["Is_Multi_Case", "kw_score"], ascending=[True, False]
        ).iterrows():
            mark   = "✦" if r["Is_Multi_Case"] else " "
            q_mark = " [Q]" if r["qwen_confirmed"] else ""
            print(f"    {mark} kw={r['kw_score']:.3f}"
                  f"  wc={r['word_count']:>6}{q_mark}"
                  f"  {r['Filename']}")

    print("\n" + "=" * 72)
    print(f"  ユニーク論文: {total_unique}本  論文×事例ペア: {total_pairs}件"
          + (f"  うちQwen補強: {n_qwen_only}件" if n_qwen_only else ""))
    print(f"  出力: {args.out}")

    # 後処理チェック
    warnings_found = False

    low_wc = df_sel[df_sel["word_count"] < MIN_WORD_COUNT]
    if not low_wc.empty:
        warnings_found = True
        print(f"\n  ⚠️  語数が少ない論文（wc < {MIN_WORD_COUNT}）— OCR再処理を検討:")
        for _, r in low_wc.iterrows():
            print(f"    wc={r['word_count']:>5}  [{r['Case_Override'][:25]}]  {r['Filename']}")

    dup_pairs: List[str] = []
    for case in PREDEFINED_CASES:
        sub = df_sel[df_sel["Case_Override"] == case].copy()
        wc_groups = sub.groupby("word_count")["Filename"].apply(list)
        for wc_val, fns in wc_groups.items():
            if len(fns) > 1 and wc_val > 0:
                dup_pairs.append(
                    f"    [{case[:25]}] wc={wc_val} → " + " / ".join(fns)
                )
    if dup_pairs:
        warnings_found = True
        print(f"\n  ⚠️  同一事例内で語数が完全一致（重複・抽出ミスの疑い）:")
        for line in dup_pairs:
            print(line)

    if not warnings_found:
        print("\n  ✓ 後処理チェック: 問題なし")

    print("=" * 72)


if __name__ == "__main__":
    main()
