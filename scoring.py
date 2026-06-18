"""
scoring.py — Phase 5a: paper-level 5因子スコアリング

【パイプライン上の位置】
  prescreen.py  → selected_papers.csv
  scoring.py    ← このスクリプト  → main_analysis_results.csv
  analyze.py    → 統計分析・可視化

【スコアリング設計】
  一次スコア: 決定論的ルールベース（段落単位キーワード密度）
  二次スコア: LLM 補助（Ollama / Qwen, temperature=0）     ← オプション
  最終スコア: rule * (1 - llm_weight) + llm * llm_weight   ← 論文単位の FS

  LLM は補助であり、最終値は一次スコアが主体。
  temperature=0 により LLM 出力も決定論的。

【出力列】
  Filename, Case, Land, Labor, Culture, Political, Economic, kw_score, word_count

【実行例】
  python3 scoring.py                   # 全論文をスコアリング
  python3 scoring.py --no-llm          # ルールベースのみ（LLMなし）
  python3 scoring.py --resume          # 既存出力をスキップして追記
  python3 scoring.py --files foo.pdf   # 特定ファイルのみ再スコアリング
"""

import os
import re
import sys
import json
import time
import argparse
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

# ---------------------------------------------------------
# パス設定
# ---------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SELECTED_CSV  = os.path.join(BASE_DIR, "selected_papers.csv")
CACHE_DIR     = os.path.join(BASE_DIR, "extraction_cache")
OUT_CSV       = os.path.join(BASE_DIR, "main_analysis_results.csv")

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
DEFAULT_MODEL   = "qwen2.5:7b"

# ---------------------------------------------------------
# 定数
# ---------------------------------------------------------
FACTORS          = ["Land", "Labor", "Culture", "Political", "Economic"]
MAX_PARAS        = 40      # 段落サンプリング上限
MIN_PARA_WORDS   = 20      # 段落として扱う最低語数
MIN_PARA_CHARS   = 80      # 段落の最低文字数

# ---------------------------------------------------------
# 5因子 × キーワード辞書（決定論的ルールベース用）
# tier1: 高識別力の固有語（1件 = 高スコア寄与）
# tier2: 広義の関連語  （複数件 = 中スコア寄与）
# ---------------------------------------------------------
FACTOR_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "Land": {
        "tier1": [
            "land dispossession", "land confiscation", "land expropriation",
            "land alienation", "land seizure", "land survey", "cadastral survey",
            "terra nullius", "allotment", "land act", "land policy",
            "eviction", "forced removal", "displacement", "resettlement",
            "土地収奪", "土地没収", "土地調査", "強制移住",
        ],
        "tier2": [
            "land", "territory", "property", "ownership", "title deed",
            "reservation", "indigenous land", "native land", "common land",
            "forest", "agricultural", "tenure", "enclosure",
            "土地", "領土", "所有", "農地",
        ],
    },
    "Labor": {
        "tier1": [
            "forced labor", "forced labour", "corvée", "bonded labor",
            "slavery", "enslaved", "indentured", "peonage",
            "tributary labor", "labor conscription", "compulsory labor",
            "coerced labor", "unfree labor", "plantation labor",
            "coolie", "poenale sanctie",
            "強制労働", "賦役", "奴隷", "徴用",
        ],
        "tier2": [
            "labor", "labour", "work", "worker", "workforce",
            "wage", "toil", "exploitation", "service", "tributary",
            "plantation", "mine", "harvest", "extraction",
            "労働", "賃金", "搾取",
        ],
    },
    "Culture": {
        "tier1": [
            "assimilation", "cultural assimilation", "forced assimilation",
            "language ban", "language suppression", "language prohibition",
            "boarding school", "residential school", "mission school",
            "christianization", "conversion", "de-indigenization",
            "kominka", "dōka", "naichi encho", "sōshi-kaimei",
            "stolen generation", "child removal",
            "文化同化", "同化政策", "言語禁止", "皇民化", "創氏改名",
        ],
        "tier2": [
            "assimilation", "culture", "language", "education", "school",
            "religion", "identity", "tradition", "custom", "dress",
            "name", "ceremony", "ritual", "indigenous knowledge",
            "文化", "言語", "教育", "宗教", "アイデンティティ",
        ],
    },
    "Political": {
        "tier1": [
            "colonial governance", "colonial administration", "colonial rule",
            "subjugation", "political subjugation", "sovereignty",
            "protectorate", "annexation", "incorporation",
            "governor-general", "colonial policy", "martial law",
            "colonial state", "colonial authority", "political control",
            "dispossession of sovereignty", "loss of sovereignty",
            "植民地統治", "政治支配", "主権剥奪", "併合", "総督府",
        ],
        "tier2": [
            "governance", "government", "administration", "policy",
            "law", "regulation", "control", "authority", "dominion",
            "empire", "imperial", "colonial", "rule", "power",
            "統治", "政策", "支配", "権力", "帝国",
        ],
    },
    "Economic": {
        "tier1": [
            "economic extraction", "resource extraction", "tribute",
            "monopoly trade", "exploitation", "plunder", "looting",
            "taxation", "tax burden", "tribute system",
            "cultuurstelsel", "cultivation system", "permanent settlement",
            "cash crop", "commodity extraction", "profit repatriation",
            "経済収奪", "資源収奪", "貢税", "専売制", "収奪",
        ],
        "tier2": [
            "trade", "commerce", "market", "economy", "revenue",
            "profit", "tax", "commodity", "resource", "export",
            "import", "company", "merchant", "finance",
            "経済", "貿易", "税", "利益", "商業",
        ],
    },
}

# ---------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------
def fmt_eta(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def count_words(text: str) -> int:
    en = len(re.findall(r"[a-zA-Z]{2,}", text))
    ja = len(re.findall(r"[\u3040-\u9FFF]{2,}", text))
    return en + ja


def load_cache_text(cache_dir: str, filename: str) -> str:
    """extraction_cache/<filename>.txt を読んで返す。"""
    path = os.path.join(cache_dir, filename + ".txt")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_paragraphs(text: str) -> List[str]:
    """
    テキストを段落に分割し、先頭・中間・末尾からサンプリングする。
    MAX_PARAS を超える場合は均等にサンプリングして上限内に収める。
    """
    segs  = re.split(r"\n\s*\n", text)
    paras = [s.strip() for s in segs
             if len(s.strip()) >= MIN_PARA_CHARS
             and count_words(s) >= MIN_PARA_WORDS]

    n = len(paras)
    if n <= MAX_PARAS:
        return paras

    mid    = n // 2
    head   = paras[:MAX_PARAS // 2]
    middle = paras[mid - MAX_PARAS // 8 : mid + MAX_PARAS // 8]
    tail   = paras[-(MAX_PARAS // 4):]

    seen, result = set(), []
    for p in head + middle + tail:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result[:MAX_PARAS]


# ---------------------------------------------------------
# ルールベーススコアリング（決定論的・一次スコア）
# ---------------------------------------------------------
def rule_score_paragraph(para: str) -> Dict[str, float]:
    """
    段落テキストに対して0–5の因子スコアを返す。
    Tier1ヒット数とTier2ヒット数の組み合わせで決定論的に算出。
    """
    para_lower = para.lower()
    scores: Dict[str, float] = {}
    for factor, kw_dict in FACTOR_KEYWORDS.items():
        t1 = sum(1 for kw in kw_dict["tier1"] if kw.lower() in para_lower)
        t2 = sum(1 for kw in kw_dict["tier2"] if kw.lower() in para_lower)

        if t1 >= 2:
            base = 4.0 + min(t1 - 2, 1) * 0.5
        elif t1 == 1:
            base = 2.5 + min(t2 * 0.2, 1.0)
        elif t2 >= 4:
            base = 2.0 + min((t2 - 4) * 0.1, 0.5)
        elif t2 >= 2:
            base = 1.0 + (t2 - 2) * 0.25
        elif t2 == 1:
            base = 0.5
        else:
            base = 0.0

        scores[factor] = round(min(base, 5.0), 3)
    return scores


def rule_score_text(paragraphs: List[str]) -> Dict[str, float]:
    """
    全段落のルールスコアを段落位置重みつきで平均し、
    論文単位（paper-level）の一次スコアを返す。
    先頭20%・末尾20%の段落は重み1.2、それ以外は1.0。
    """
    if not paragraphs:
        return {f: 0.0 for f in FACTORS}

    n            = len(paragraphs)
    total        = defaultdict(float)
    total_weight = 0.0

    for i, para in enumerate(paragraphs):
        rel_pos = i / max(n - 1, 1)
        weight  = 1.2 if rel_pos < 0.2 or rel_pos > 0.8 else 1.0
        s       = rule_score_paragraph(para)
        for f in FACTORS:
            total[f] += s[f] * weight
        total_weight += weight

    return {f: round(total[f] / total_weight, 3) for f in FACTORS}


# ---------------------------------------------------------
# LLM スコアリング補助（二次スコア・オプション）
# ---------------------------------------------------------
SCORING_SYSTEM = (
    "You are a historian specializing in comparative colonial studies. "
    "Score the paragraph on five colonial governance dimensions (0–5 each). "
    "0=absent, 1-2=briefly mentioned, 3-4=substantially discussed, 5=primary focus. "
    "Return ONLY a JSON object: {\"Land\":N,\"Labor\":N,\"Culture\":N,\"Political\":N,\"Economic\":N}. "
    "No other text."
)


def check_ollama(model: str) -> bool:
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        return any(model.split(":")[0] in m for m in models)
    except Exception:
        return False


def llm_score_paragraph(
    para: str, model: str, timeout: int = 60, retries: int = 1,
) -> Optional[Dict[str, float]]:
    """1段落を LLM でスコアリング。失敗時は None を返す。"""
    payload = {
        "model":  model,
        "prompt": f"{SCORING_SYSTEM}\n\nParagraph:\n{para[:1200]}\n\nJSON:",
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "think": False, "num_predict": 128},
    }
    for attempt in range(retries + 1):
        try:
            r   = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=timeout)
            r.raise_for_status()
            raw = r.json().get("response", "")
            raw = re.sub(r"<think>[\s\S]*?</think>", "", raw,
                         flags=re.IGNORECASE).strip()
            m   = re.search(r"\{[\s\S]*\}", raw)
            if m:
                data   = json.loads(m.group())
                result = {}
                for f in FACTORS:
                    v = data.get(f, data.get(f.lower(), 0))
                    result[f] = float(max(0, min(5, v)))
                return result
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(1)
        except Exception:
            pass
    return None


def llm_score_text(
    paragraphs: List[str], model: str,
) -> Optional[Dict[str, float]]:
    """
    段落をサンプリングして LLM でスコアリングし、段落平均を返す。
    最大20段落を先頭・中間・末尾から均等に取得する。
    """
    n      = len(paragraphs)
    sample = (paragraphs[:8]
              + paragraphs[n//2-4:n//2+4]
              + paragraphs[-8:])[:20] if n > 20 else paragraphs

    scored = [llm_score_paragraph(p, model) for p in sample]
    scored = [s for s in scored if s is not None]
    if not scored:
        return None

    return {f: round(sum(s[f] for s in scored) / len(scored), 3)
            for f in FACTORS}


# ---------------------------------------------------------
# ハイブリッドスコア合成
# ---------------------------------------------------------
def hybrid_score(
    rule: Dict[str, float],
    llm: Optional[Dict[str, float]],
    llm_weight: float,
) -> Dict[str, float]:
    """
    最終スコア = rule * (1 - llm_weight) + llm * llm_weight
    LLM が None の場合（失敗・--no-llm 指定）はルールスコアをそのまま返す。
    """
    if llm is None:
        return rule
    return {
        f: round(rule[f] * (1 - llm_weight) + llm[f] * llm_weight, 3)
        for f in FACTORS
    }


# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5a: selected_papers.csv → main_analysis_results.csv"
    )
    parser.add_argument("--selected-csv", default=SELECTED_CSV, metavar="FILE",
                        help=f"選抜論文リスト（デフォルト: {SELECTED_CSV}）")
    parser.add_argument("--cache",        default=CACHE_DIR,    metavar="DIR",
                        help=f"テキストキャッシュ（デフォルト: {CACHE_DIR}）")
    parser.add_argument("--out",          default=OUT_CSV,      metavar="FILE",
                        help=f"出力CSVパス（デフォルト: {OUT_CSV}）")
    parser.add_argument("--model",        default=DEFAULT_MODEL, metavar="MODEL",
                        help=f"Ollamaモデル名（デフォルト: {DEFAULT_MODEL}）")
    parser.add_argument("--no-llm",       action="store_true",
                        help="LLMを使わずルールベースのみで出力する")
    parser.add_argument("--llm-weight",   type=float, default=0.35, metavar="FLOAT",
                        help="LLMスコアの重み 0.0–1.0（デフォルト: 0.35）")
    parser.add_argument("--resume",       action="store_true",
                        help="既存の出力CSVにある論文をスキップして追記する")
    parser.add_argument("--files",        nargs="+", metavar="FILENAME",
                        help="再スコアリングするファイル名を個別指定")
    args = parser.parse_args()

    print("=" * 72)
    print("  scoring.py — Phase 5a 論文スコアリング")
    print(f"  LLM={'なし（ルールベースのみ）' if args.no_llm else args.model}"
          f"  llm_weight={args.llm_weight}"
          f"  resume={args.resume}")
    print("=" * 72)

    # --- selected_papers.csv の読み込み ---
    if not os.path.isfile(args.selected_csv):
        print(f"  ❌ {args.selected_csv} が見つかりません。prescreen.py を先に実行してください。")
        sys.exit(1)

    df_sel = pd.read_csv(args.selected_csv, encoding="utf-8-sig")
    n_total = len(df_sel)
    print(f"\n  選抜論文: {n_total} 件 "
          f"（ユニーク: {df_sel['Filename'].nunique()}本 "
          f"× 事例: {df_sel['Case_Override'].nunique()}事例）\n")

    # --- --files 指定があれば絞り込み ---
    if args.files:
        df_sel = df_sel[df_sel["Filename"].isin(args.files)]
        print(f"  --files 指定: {len(df_sel)} 件に絞り込み\n")

    # --- --resume: 既存スコアをロードしてスキップリストを作成 ---
    done_keys: set = set()   # (Filename, Case_Override) の済みペア
    existing_rows: List[dict] = []

    if args.resume and os.path.isfile(args.out):
        df_existing = pd.read_csv(args.out, encoding="utf-8-sig")
        for _, row in df_existing.iterrows():
            done_keys.add((row["Filename"], row["Case"]))
            existing_rows.append(row.to_dict())
        print(f"  --resume: 既存スコア {len(done_keys)} 件をスキップします\n")

    # --- Ollama 疎通確認 ---
    use_llm = not args.no_llm
    if use_llm:
        if check_ollama(args.model):
            print(f"  Ollama: 接続OK  model={args.model}\n")
        else:
            print(f"  Ollama: 接続失敗 → ルールベースのみで実行します\n")
            use_llm = False

    # --- スコアリングループ ---
    results: List[dict] = list(existing_rows)
    times:   List[float] = []
    skipped = 0
    failed  = 0

    for idx, row in df_sel.iterrows():
        filename  = str(row["Filename"]).strip()
        case      = str(row["Case_Override"]).strip()
        kw_score  = float(row.get("kw_score", 0.0) or 0.0)
        wc_stored = int(row.get("word_count", 0) or 0)

        # resume スキップ
        if (filename, case) in done_keys:
            skipped += 1
            continue

        t0   = time.time()
        text = load_cache_text(args.cache, filename)
        wc   = count_words(text) if text else wc_stored

        if not text:
            print(f"  [{idx+1:3d}/{n_total}] ⚠️  テキストなし: {filename[:60]}")
            failed += 1
            results.append({
                "Filename": filename, "Case": case,
                **{f: 0.0 for f in FACTORS},
                "kw_score": kw_score, "word_count": wc,
                "score_method": "no_text",
            })
            continue

        # 段落分割
        paragraphs = split_paragraphs(text)

        # 一次スコア（ルールベース・決定論的）
        rule = rule_score_text(paragraphs)

        # 二次スコア（LLM 補助・オプション）
        llm = llm_score_text(paragraphs, args.model) if use_llm else None

        # ハイブリッド合成 → 最終 paper-level FS
        final = hybrid_score(rule, llm, args.llm_weight)

        method = "hybrid" if llm else "rule"
        results.append({
            "Filename":    filename,
            "Case":        case,
            **final,
            "kw_score":    kw_score,
            "word_count":  wc,
            "score_method": method,
        })
        done_keys.add((filename, case))

        dt = time.time() - t0
        times.append(dt)
        avg = sum(times) / len(times)
        remaining = n_total - (idx + 1) - skipped
        eta = avg * max(remaining, 0)

        scores_str = "  ".join(f"{f[0]}={final[f]:.2f}" for f in FACTORS)
        print(f"  [{idx+1:3d}/{n_total}] [{method}] {filename[:45]:<45}\n"
              f"    {scores_str}  ETA={fmt_eta(eta)}")

    # --- CSV 書き出し ---
    out_cols = ["Filename", "Case"] + FACTORS + ["kw_score", "word_count", "score_method"]
    df_out = pd.DataFrame(results)[out_cols]
    df_out.to_csv(args.out, index=False, encoding="utf-8-sig")

    # --- サマリー ---
    elapsed = sum(times)
    print("\n" + "=" * 72)
    print(f"  完了: {len(results)} 件  スキップ: {skipped}  テキストなし: {failed}")
    print(f"  処理時間: {fmt_eta(elapsed)}")
    print(f"  出力: {args.out}")
    print(f"\n  次のステップ:")
    print(f"    python3 analyze.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
