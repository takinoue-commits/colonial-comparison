from typing import DefaultDict, Dict, List, Optional, Set, Tuple
# =========================================================
# recompute_kw_screen.py
# キャッシュ済みテキストからキーワードスコアを再計算する
#
# 【用途】
#   prescreen.py を実行する前にキーワードスコアの分布を確認・
#   デバッグしたい場合や、BUILTIN_KW を更新した後に全件を
#   再スコアリングしたい場合に使う補助スクリプト。
#
# 【キャッシュ形式】pdf_extract.py の出力と一致させること
#   extraction_cache/<filename>.pdf.txt   ← 本文テキスト
#   extraction_cache/<filename>.pdf.meta  ← {"page_count": N}
#
# 【出力 CSV 列の互換性】
#   kw_score_<case>  : 0.0–1.0 の正規化スコア（prescreen.py 互換）
#   governance_density : 統治語密度（prescreen.py と同じ計算式）
#   top_case         : 最高スコアの事例名
#   assigned_cases   : RELEVANCE_THRESHOLD 以上の事例（| 区切り）
#
# 【実行例】
#   python3 recompute_kw_screen.py
#   python3 recompute_kw_screen.py --threshold 0.20
#   python3 recompute_kw_screen.py --cache ./extraction_cache --out my_report.csv
# =========================================================

import os
import re
import json
import time
import argparse
import sys
from datetime import timedelta

import pandas as pd

# ---------------------------------------------------------
# 定数
# ---------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "extraction_cache")
OUT_CSV   = os.path.join(BASE_DIR, "screening_report_anchor_kw.csv")

RELEVANCE_THRESHOLD = 0.15   # prescreen.py と統一
MAX_CASES_PER_FILE  = 2      # prescreen.py と統一

# ---------------------------------------------------------
# キーワード辞書（prescreen.py の CASE_KEYWORDS と整合）
# tier1: 固有の識別力が高い語 → 1ヒットで高スコア
# tier2: 広義の関連語         → 複数ヒットで中スコア
# ---------------------------------------------------------
BUILTIN_KW: Dict[str, Dict[str, List[str]]] = {
    "Ainu (Edo Period - Basho Ukeoi)": {
        "tier1": [
            "basho ukeoi", "ezochi", "蝦夷地", "場所請負",
            "matsumae", "shakushain", "松前藩",
        ],
        "tier2": [
            "ainu", "アイヌ", "ezo", "蝦夷",
            "tokugawa", "edo", "japanese indigenous",
        ],
    },
    "Ainu (Meiji Period - Former Aborigine Law)": {
        "tier1": [
            "former aborigine", "旧土人", "former aborigines protection",
            "hokkaido colonization", "北海道開拓", "アイヌ保護",
        ],
        "tier2": [
            "ainu", "アイヌ", "assimilation", "同化", "meiji",
            "hokkaido", "indigenous hokkaido",
        ],
    },
    "Aboriginal Australians": {
        "tier1": [
            "terra nullius", "stolen generations", "mabo decision",
            "aboriginal protection act",
        ],
        "tier2": [
            "aboriginal", "aborigines", "torres strait islander",
            "indigenous australian", "first nations australia",
        ],
    },
    "Maori (New Zealand)": {
        "tier1": [
            "waitangi", "aotearoa", "raupatu", "new zealand land wars",
        ],
        "tier2": [
            "maori", "māori", "iwi", "new zealand indigenous",
            "tino rangatiratanga",
        ],
    },
    "Native American (US)": {
        "tier1": [
            "dawes act", "indian removal act", "bureau of indian affairs",
            "trail of tears",
        ],
        "tier2": [
            "native american", "american indian", "tribal", "reservation",
            "indigenous american", "federal indian policy",
        ],
    },
    "Taiwan (Japanese Rule)": {
        "tier1": [
            "governor-general of taiwan", "kominka", "musha incident",
            "台湾総督府", "臺灣",
        ],
        "tier2": [
            "taiwan", "台湾", "japanese colonial", "日本統治",
            "taiwanese colonial",
        ],
    },
    "Korea (Japanese Rule)": {
        "tier1": [
            "annexation of korea", "韓国併合", "land survey project korea",
            "government-general korea", "march first movement",
        ],
        "tier2": [
            "korea", "朝鮮", "korean colonial", "japanese colonial korea",
        ],
    },
    "Indonesia (Dutch East Indies)": {
        "tier1": [
            "cultuurstelsel", "dutch east indies", "poenale sanctie",
            "herendiensten", "voc",
        ],
        "tier2": [
            "indonesia", "java", "javanese", "dutch colonial",
            "cultivation system",
        ],
    },
    "Bengal (British India)": {
        "tier1": [
            "permanent settlement", "zamindari", "plassey", "nawab bengal",
            "bengal famine",
        ],
        "tier2": [
            "bengal", "bengali", "east india company", "british raj",
            "colonial india",
        ],
    },
    "Ireland": {
        "tier1": [
            "great famine", "plantation ireland", "penal laws",
            "cromwell ireland", "ulster plantation",
        ],
        "tier2": [
            "ireland", "irish", "colonial ireland", "british ireland",
        ],
    },
    "Ryukyu (Okinawa)": {
        "tier1": [
            "ryukyu disposition", "琉球処分", "satsuma domain ryukyu",
            "disposition of ryukyu",
        ],
        "tier2": [
            "ryukyu", "琉球", "okinawa", "沖縄", "ryukyuan",
        ],
    },
}

CASES = list(BUILTIN_KW.keys())

# prescreen.py の GOVERNANCE_KEYWORDS と同じリスト
GOVERNANCE_KEYWORDS = [
    "land dispossession", "land confiscation", "forced labor", "forced labour",
    "corvée", "cultural assimilation", "language ban", "colonial governance",
    "economic extraction", "resource extraction", "political subjugation",
    "土地収奪", "強制労働", "文化同化", "政治支配", "経済収奪",
    "colonial", "colonialism", "colonization", "settler", "indigenous",
    "subjugation", "domination", "exploitation", "tributary", "trade system",
    "assimilation", "dispossession", "displacement", "occupation", "sovereignty",
    "ainu", "matsumae", "basho", "ezochi",
]

# ---------------------------------------------------------
# スコアリング（prescreen.py の score_case と同じロジック）
# ---------------------------------------------------------
def score_case(text: str, case: str) -> float:
    """
    Tier1/Tier2 二層スコアリング。
    返値は 0.0–1.0 の浮動小数点（prescreen.py 互換）。
    """
    kw_dict = BUILTIN_KW.get(case, {})
    tier1   = kw_dict.get("tier1", [])
    tier2   = kw_dict.get("tier2", [])
    if not tier1 and not tier2:
        return 0.0

    text_lower = text.lower()
    t1_hits = sum(1 for kw in tier1 if kw.lower() in text_lower)
    t2_hits = sum(1 for kw in tier2 if kw.lower() in text_lower)

    # 段落サンプリング（先頭15 + 中間10 + 末尾10）
    segs  = re.split(r'\n\s*\n', text)
    paras = [s.lower() for s in segs if len(s.strip().split()) >= 15]
    n     = len(paras)
    if n <= 35:
        sample = paras
    else:
        mid    = n // 2
        sample = paras[:15] + paras[mid - 5 : mid + 5] + paras[-10:]
    if not sample:
        sample = [text_lower[:2000]]

    p_t1 = sum(1 for p in sample if any(kw.lower() in p for kw in tier1))
    p_t2 = sum(1 for p in sample if any(kw.lower() in p for kw in tier2))
    sn   = max(len(sample), 1)

    if t1_hits >= 1:
        base  = 0.40 + min(t1_hits * 0.08, 0.40)
        score = min(0.90, base + (p_t1 / sn) * 0.20)
    elif t2_hits >= 3:
        base  = 0.15 + min(t2_hits * 0.025, 0.20)
        score = min(0.45, base + (p_t2 / sn) * 0.10)
    else:
        score = 0.0
    return round(score, 3)


def compute_governance_density(text: str) -> float:
    """テキスト先頭3000語の統治語密度を返す（prescreen.py と同じ計算）。"""
    head = " ".join(text.split()[:3000]).lower()
    hits = sum(1 for kw in GOVERNANCE_KEYWORDS if kw in head)
    return round(hits / len(GOVERNANCE_KEYWORDS), 3)


def assign_cases(
    scores: Dict[str, float],
    threshold: float,
    max_cases: int,
) -> List[str]:
    """スコア上位から threshold 以上の事例を最大 max_cases 件返す。"""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [case for case, s in ranked if s >= threshold][:max_cases]


# ---------------------------------------------------------
# キャッシュ読み込み（pdf_extract.py 更新版と互換）
# ---------------------------------------------------------
def load_cache_text(cache_dir: str, filename: str) -> Tuple[str, int]:
    """
    <filename>.txt と <filename>.meta を読む。
    filename は "foo.pdf" のような元ファイル名（拡張子込み）。
    戻り値: (text, page_count)
    """
    txt_path  = os.path.join(cache_dir, filename + ".txt")
    meta_path = os.path.join(cache_dir, filename + ".meta")

    if not os.path.exists(txt_path):
        return "", 0

    with open(txt_path, encoding="utf-8") as f:
        text = f.read()

    page_count = 0
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        page_count = meta.get("page_count", 0)

    return text, page_count


def list_cached_files(cache_dir: str) -> List[str]:
    """
    キャッシュディレクトリから処理済みファイル名（"foo.pdf" 形式）を列挙する。
    .txt ファイルが存在することを基準とする。
    """
    txt_files = [
        f[:-4]  # ".txt" を除去して "foo.pdf" を得る
        for f in os.listdir(cache_dir)
        if f.endswith(".txt")
    ]
    return sorted(txt_files)


# ---------------------------------------------------------
# ETA 表示
# ---------------------------------------------------------
def fmt_eta(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


# ---------------------------------------------------------
# メイン
# ---------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="キャッシュ済みテキストからKWスコアを再計算する"
    )
    parser.add_argument(
        "--cache", default=CACHE_DIR, metavar="DIR",
        help=f"キャッシュディレクトリ（デフォルト: {CACHE_DIR}）",
    )
    parser.add_argument(
        "--out", default=OUT_CSV, metavar="FILE",
        help=f"出力CSVパス（デフォルト: {OUT_CSV}）",
    )
    parser.add_argument(
        "--threshold", type=float, default=RELEVANCE_THRESHOLD,
        metavar="FLOAT",
        help=f"事例割当の最低スコア（デフォルト: {RELEVANCE_THRESHOLD}）",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.cache):
        print(f"エラー: キャッシュディレクトリが見つかりません: {args.cache}")
        sys.exit(1)

    filenames = list_cached_files(args.cache)
    total     = len(filenames)

    if total == 0:
        print(f"キャッシュ (.txt) が見つかりません: {args.cache}")
        print("先に pdf_extract.py を実行してください。")
        sys.exit(0)

    print("=" * 72)
    print(" recompute_kw_screen.py — KW スコア再計算")
    print(f" 対象ファイル数 : {total}")
    print(f" スコア閾値     : {args.threshold}")
    print(f" キャッシュ     : {args.cache}")
    print("=" * 72)

    rows       = []
    times      = []
    start_all  = time.time()
    no_text    = []
    no_case    = []

    for i, filename in enumerate(filenames, start=1):
        t0 = time.time()

        text, page_count = load_cache_text(args.cache, filename)

        if not text.strip():
            no_text.append(filename)
            print(f"  [{i:3d}/{total}] ✗ テキストなし: {filename}")
            continue

        # スコア計算
        scores = {case: score_case(text, case) for case in CASES}
        gov_density     = compute_governance_density(text)
        assigned        = assign_cases(scores, args.threshold, MAX_CASES_PER_FILE)
        top_case        = max(scores, key=scores.get) if scores else "Unknown"
        top_score       = scores[top_case]

        if not assigned:
            no_case.append(filename)

        # 行データ組み立て
        row: dict = {
            "Filename":           filename,
            "page_count":         page_count,
            "governance_density": gov_density,
            "top_case":           top_case,
            "top_score":          top_score,
            "assigned_cases":     " | ".join(assigned) if assigned else "",
        }
        for case in CASES:
            row[f"kw_score_{case}"] = scores[case]

        rows.append(row)

        dt  = time.time() - t0
        times.append(dt)
        avg = sum(times) / len(times)
        eta = avg * (total - i)

        # 上位2事例をログに表示
        top2 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
        top2_str = "  ".join(f"{c[:22]}={s:.2f}" for c, s in top2 if s > 0)
        print(
            f"  [{i:3d}/{total}] {filename[:55]:<55}\n"
            f"    {top2_str or '（関連事例なし）'}"
            f"  gov={gov_density:.3f}  ETA={fmt_eta(eta)}"
        )

    # CSV 出力
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    # サマリー
    elapsed = time.time() - start_all
    print("\n" + "=" * 72)
    print(f"  完了: {len(rows)} 件  テキストなし: {len(no_text)}  事例割当なし: {len(no_case)}")
    print(f"  処理時間: {fmt_eta(elapsed)}")
    print(f"  出力: {args.out}")

    if no_text:
        print(f"\n  ⚠️  テキスト抽出が必要なファイル（{len(no_text)} 件）:")
        for fn in no_text:
            print(f"    python3 pdf_extract.py --files \"{fn}\" --force")

    if no_case:
        print(f"\n  ⚠️  事例割当なし（全スコア閾値未満）: {len(no_case)} 件")
        for fn in no_case[:10]:
            print(f"    {fn}")
        if len(no_case) > 10:
            print(f"    ... 他 {len(no_case) - 10} 件")

    print("=" * 72)

    # スコア分布サマリーを表示
    if rows:
        print("\n  【事例別 閾値到達件数（threshold={:.2f}）】".format(args.threshold))
        for case in CASES:
            col     = f"kw_score_{case}"
            n_above = int((df[col] >= args.threshold).sum())
            bar     = "█" * n_above + "░" * (total - len(no_text) - n_above)
            print(f"  {n_above:3d}  {bar[:40]}  {case}")


if __name__ == "__main__":
    main()