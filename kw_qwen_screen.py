# ======================================================
# kw_qwen_screen.py
# LLM（Qwen / Ollama）による事例スクリーニング補助スクリプト
#
# 【位置づけ】
#   1. pdf_extract.py       → extraction_cache/*.pdf.txt を生成
#   2. recompute_kw_screen.py → screening_report_anchor_kw.csv を生成
#   3. kw_qwen_screen.py    ← このスクリプト（LLMで判定を補強）
#   4. prescreen.py         → 最終的な論文選定
#
# 【出力 CSV 列】
#   Filename, qwen_case_1, qwen_case_2, qwen_confidence, kw_consistent
#   ※ qwen_case_* は prescreen.py の PREDEFINED_CASES と同じ完全名を使用
#
# 【実行例】
#   python3 kw_qwen_screen.py
#   python3 kw_qwen_screen.py --model qwen2.5:7b --max-chars 3000
#   python3 kw_qwen_screen.py --files foo.pdf bar.pdf
# ======================================================

import os
import re
import json
import csv
import time
import argparse
import sys
from datetime import timedelta
from typing import Dict, List, Optional

import requests

# ------------------------------------------------------
# パス設定（他スクリプトと統一）
# ------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(BASE_DIR, "extraction_cache")
KW_CSV     = os.path.join(BASE_DIR, "screening_report_anchor_kw.csv")
OUT_CSV    = os.path.join(BASE_DIR, "screening_report_qwen.csv")

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
DEFAULT_MODEL   = "qwen2.5:7b"

# prescreen.py の RELEVANCE_THRESHOLD と統一
RELEVANCE_THRESHOLD = 0.15

# 事例ごとのKW閾値（LLMに渡すかどうかの判定）
# 前回の割当件数が少なかった事例は閾値を下げてLLMに届く論文を増やす
# Indonesia=0, NativeAmerican=1, Aboriginal=1, Bengal=2, Ryukyu=2 が問題
CASE_KW_THRESHOLD: Dict[str, float] = {
    "Ainu (Edo Period - Basho Ukeoi)":            0.15,
    "Ainu (Meiji Period - Former Aborigine Law)":  0.15,
    "Maori (New Zealand)":                         0.15,
    "Native American (US)":                        0.08,  # 低め: 論文不足
    "Aboriginal Australians":                      0.08,  # 低め: 論文不足
    "Taiwan (Japanese Rule)":                      0.20,  # 高め: 過剰割当を抑制
    "Korea (Japanese Rule)":                       0.15,
    "Indonesia (Dutch East Indies)":               0.05,  # 最低: 0件問題を解消
    "Bengal (British India)":                      0.10,  # 低め: 論文不足
    "Ireland":                                     0.15,
    "Ryukyu (Okinawa)":                            0.10,  # 低め: 論文不足
}

# ------------------------------------------------------
# 事例リスト（prescreen.py の PREDEFINED_CASES と完全一致）
# ここを変更する場合は prescreen.py も同期すること
# ------------------------------------------------------
PREDEFINED_CASES = [
    "Ainu (Edo Period - Basho Ukeoi)",
    "Ainu (Meiji Period - Former Aborigine Law)",
    "Maori (New Zealand)",
    "Native American (US)",
    "Aboriginal Australians",
    "Taiwan (Japanese Rule)",
    "Korea (Japanese Rule)",
    "Indonesia (Dutch East Indies)",
    "Bengal (British India)",
    "Ireland",
    "Ryukyu (Okinawa)",
]

# プロンプト内でのみ使う短縮ラベル → 完全名 のマッピング
# （LLM に長い括弧付き名称を返させると解析が不安定なため）
SHORT_TO_FULL: dict[str, str] = {
    "Ainu_Edo":       "Ainu (Edo Period - Basho Ukeoi)",
    "Ainu_Meiji":     "Ainu (Meiji Period - Former Aborigine Law)",
    "Maori":          "Maori (New Zealand)",
    "NativeAmerican": "Native American (US)",
    "Aboriginal":     "Aboriginal Australians",
    "Taiwan":         "Taiwan (Japanese Rule)",
    "Korea":          "Korea (Japanese Rule)",
    "Indonesia":      "Indonesia (Dutch East Indies)",
    "Bengal":         "Bengal (British India)",
    "Ireland":        "Ireland",
    "Ryukyu":         "Ryukyu (Okinawa)",
    "None":           "",
}
SHORT_LABELS = list(SHORT_TO_FULL.keys())  # プロンプトに渡すリスト

# ------------------------------------------------------
# Ollama 疎通確認
# ------------------------------------------------------
def check_ollama(model: str) -> bool:
    """Ollama が起動していて指定モデルが利用可能かを確認する。"""
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=5)
        if r.status_code != 200:
            return False
        models = [m["name"] for m in r.json().get("models", [])]
        base   = model.split(":")[0]
        return any(base in m for m in models)
    except Exception:
        return False


# ------------------------------------------------------
# LLM 呼び出し（リトライ付き）
# ------------------------------------------------------
def call_ollama(prompt: str, model: str, timeout: int = 90,
                retries: int = 2) -> str:
    """
    Ollama にリクエストを送り、レスポンス文字列を返す。
    失敗時は空文字を返す。
    """
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "think":       False,   # Qwen3 Thinking モードを無効化
            "num_predict": 64,      # 短いラベル返答なので小さく抑える
        },
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=timeout)
            r.raise_for_status()
            raw = r.json().get("response", "")
            # <think>...</think> ブロックを除去
            raw = re.sub(r"<think>[\s\S]*?</think>", "", raw,
                         flags=re.IGNORECASE).strip()
            return raw
        except requests.exceptions.Timeout:
            if attempt < retries:
                print(f"    [警告] タイムアウト({timeout}s)、リトライ {attempt+1}/{retries}")
                time.sleep(2)
            else:
                print("    [エラー] Ollama タイムアウト（スキップ）")
        except requests.exceptions.ConnectionError:
            print("    [エラー] Ollama に接続できません（`ollama serve` を確認）")
            break
        except Exception as e:
            print(f"    [エラー] {e}")
            break
    return ""


# ------------------------------------------------------
# プロンプト構築・レスポンス解析
# ------------------------------------------------------
def build_prompt(text: str, max_chars: int) -> str:
    """
    最大 max_chars 文字のテキストを使ってプロンプトを組み立てる。
    単語途中でのカットを避けるため空白境界で切り捨てる。
    """
    if len(text) > max_chars:
        cut = text[:max_chars].rsplit(" ", 1)[0]
    else:
        cut = text

    cases_str = "\n".join(f"- {c}" for c in SHORT_LABELS)

    return (
        "You are an expert in comparative colonial history.\n\n"
        "From the document below, identify up to TWO cases that are "
        "PRIMARILY and SUBSTANTIVELY discussed.\n"
        "If only one case is primary, return one label.\n"
        "If no specific case is central (general theory, methodology, etc.), "
        "return 'None'.\n\n"
        "=== GENERAL RULES ===\n"
        "1. Choose the case(s) MOST extensively discussed, not just mentioned.\n"
        "2. Cross-case comparative papers: list both cases if both are central.\n"
        "3. General colonial theory with no specific empirical case → None.\n\n"
        "=== CASE-SPECIFIC DISAMBIGUATION RULES ===\n"
        "Ainu_Edo   : Paper focuses on Matsumae domain, basho ukeoi fishery system, "
        "Shakushain's revolt, or Ainu-wajin trade in the Edo period (pre-1868). "
        "Do NOT assign if the paper mainly covers Meiji-era policies.\n"
        "Ainu_Meiji : Paper focuses on Former Aborigines Protection Act (1899), "
        "Hokkaido colonization commission (kaitakushi), Meiji-era assimilation, "
        "or post-1868 Ainu policy. Do NOT assign if the paper mainly covers Edo-period trade.\n"
        "Indonesia  : Paper focuses on Dutch East Indies / VOC / cultuurstelsel / "
        "Java cultivation system / poenale sanctie. "
        "Assign Indonesia even if 'colonial' or 'indigenous' appear without Dutch/Java specifics, "
        "AS LONG AS Dutch or Indonesian context is clearly present.\n"
        "Taiwan     : Paper focuses specifically on Japanese colonial Taiwan, "
        "governor-general, kominka, or Musha incident. "
        "Do NOT assign Taiwan to papers about East Asian colonialism in general, "
        "or papers where Taiwan is merely mentioned alongside Korea or Ryukyu.\n"
        "Korea      : Paper focuses on Japanese colonial Korea, annexation (1910), "
        "land survey, comfort women, or March First Movement. "
        "Distinguish from papers about Korea's post-colonial period.\n"
        "NativeAmerican : Paper focuses on US federal Indian policy, tribal sovereignty, "
        "reservations, Dawes Act, or boarding schools. "
        "Assign NativeAmerican even if framed as 'settler colonialism' theory, "
        "AS LONG AS Native American/US Indigenous peoples are the empirical focus.\n"
        "Aboriginal : Paper focuses on Aboriginal Australians, terra nullius, "
        "stolen generations, Mabo decision, or Australian protection acts. "
        "Assign Aboriginal even if framed as 'settler colonialism' theory, "
        "AS LONG AS Australian Indigenous peoples are the empirical focus.\n"
        "Maori      : Paper focuses on Maori people, Treaty of Waitangi, "
        "Aotearoa/New Zealand land wars, or raupatu (land confiscation).\n"
        "Bengal     : Paper focuses on Bengal under British East India Company or Raj, "
        "permanent settlement, zamindari, or Bengal famine.\n"
        "Ireland    : Paper focuses on Irish colonial history under British rule, "
        "Great Famine, plantation system, or penal laws.\n"
        "Ryukyu     : Paper focuses on Ryukyu Kingdom / Okinawa under Japanese rule, "
        "ryukyu disposition (1879), or Satsuma domain control.\n\n"
        f"Available labels:\n{cases_str}\n\n"
        f"Document:\n{cut}\n\n"
        "Answer with ONE or TWO labels, comma-separated (e.g. 'Ainu_Edo' or "
        "'Korea, Taiwan'). No other text."
    )


def parse_response(raw: str) -> list[str]:
    """
    LLM レスポンスから有効な短縮ラベルを最大2件抽出する。
    返値は完全名のリスト（例: ["Ainu (Edo Period - Basho Ukeoi)"]）。
    """
    if not raw:
        return []

    # カンマ・改行・スラッシュで分割して短縮ラベルを探す
    tokens = re.split(r"[,\n/]+", raw)
    found  = []
    for token in tokens:
        label = token.strip().strip("'\"")
        if label in SHORT_TO_FULL:
            full = SHORT_TO_FULL[label]
            if full and full not in found:
                found.append(full)
        if len(found) >= 2:
            break
    return found


# ------------------------------------------------------
# キャッシュ読み込み（pdf_extract.py 更新版と互換）
# ------------------------------------------------------
def load_cache_text(cache_dir: str, filename: str) -> str:
    """
    filename は "foo.pdf" 形式（拡張子込み）。
    extraction_cache/foo.pdf.txt を読んで返す。
    """
    txt_path = os.path.join(cache_dir, filename + ".txt")
    if not os.path.exists(txt_path):
        return ""
    with open(txt_path, encoding="utf-8") as f:
        return f.read()


def list_cached_files(cache_dir: str) -> list[str]:
    """.txt キャッシュが存在するファイル名（"foo.pdf"）を返す。"""
    return sorted(
        f[:-4]  # ".txt" を除去
        for f in os.listdir(cache_dir)
        if f.endswith(".txt")
    )


# ------------------------------------------------------
# KWスコア CSV の読み込み（BOM・列名ゆらぎに対応）
# ------------------------------------------------------
def load_kw_scores(kw_csv: str) -> dict[str, dict]:
    """
    Filename → {列名: 値} の辞書を返す。
    ファイルが存在しない場合は空辞書を返す。
    """
    if not os.path.exists(kw_csv):
        print(f"  [警告] KWスコアCSVが見つかりません: {kw_csv}")
        print("  先に recompute_kw_screen.py を実行してください。")
        return {}

    result = {}
    with open(kw_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        # 列名のゆらぎを吸収（Filename / filename / File）
        fname_col = next(
            (c for c in fields if c.lower() in ("filename", "file")), None
        )
        if fname_col is None:
            raise RuntimeError(
                f"Filename列が見つかりません。列: {fields}")

        for row in reader:
            result[row[fname_col]] = dict(row)
    return result


# ------------------------------------------------------
# KWスコアとの整合性チェック
# ------------------------------------------------------
def kw_consistent(case_full: str, kw_row: Optional[dict]) -> Optional[bool]:
    """
    LLM が返した事例のKWスコアが事例別閾値以上かを返す。
    kw_row が None（KWスコアCSV未使用）の場合は None を返す。
    """
    if kw_row is None:
        return None
    col       = f"kw_score_{case_full}"
    score     = float(kw_row.get(col, 0.0) or 0.0)
    threshold = CASE_KW_THRESHOLD.get(case_full, RELEVANCE_THRESHOLD)
    return score >= threshold


# ------------------------------------------------------
# ETA
# ------------------------------------------------------
def fmt_eta(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


# ------------------------------------------------------
# メイン
# ------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ollama Qwen による事例スクリーニング（LLM補強）"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"使用する Ollama モデル（デフォルト: {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--max-chars", type=int, default=4000, metavar="N",
        help="LLMに渡すテキストの最大文字数（デフォルト: 4000）",
    )
    parser.add_argument(
        "--files", nargs="+", metavar="FILENAME",
        help="処理対象を絞る場合にファイル名を指定（例: foo.pdf bar.pdf）",
    )
    parser.add_argument(
        "--cache", default=CACHE_DIR, metavar="DIR",
        help=f"キャッシュディレクトリ（デフォルト: {CACHE_DIR}）",
    )
    parser.add_argument(
        "--kw-csv", default=KW_CSV, metavar="FILE",
        help=f"KWスコアCSVパス（デフォルト: {KW_CSV}）",
    )
    parser.add_argument(
        "--out", default=OUT_CSV, metavar="FILE",
        help=f"出力CSVパス（デフォルト: {OUT_CSV}）",
    )
    args = parser.parse_args()

    # Ollama 疎通確認
    print("=" * 68)
    print(f"  kw_qwen_screen.py  model={args.model}")
    print("=" * 68)

    if not check_ollama(args.model):
        print(f"  [エラー] Ollama に接続できないか、モデル '{args.model}' が見つかりません。")
        print("  `ollama serve` と `ollama pull {args.model}` を確認してください。")
        sys.exit(1)
    print(f"  Ollama: 接続OK  model={args.model}\n")

    # 対象ファイルの決定
    if args.files:
        all_files = args.files
    else:
        all_files = list_cached_files(args.cache)

    if not all_files:
        print("処理対象のキャッシュファイルがありません。")
        sys.exit(0)

    # KWスコア読み込み（任意）
    kw_scores = load_kw_scores(args.kw_csv)

    total    = len(all_files)
    rows     = []
    times    = []
    skipped  = []

    print(f"  対象: {total} 件\n")

    for i, filename in enumerate(all_files, start=1):
        t0   = time.time()
        text = load_cache_text(args.cache, filename)

        if not text.strip():
            skipped.append(filename)
            print(f"  [{i:3d}/{total}] ✗ テキストなし（スキップ）: {filename}")
            continue

        # 事例別KW閾値を用いて、この論文がいずれかの事例候補になりえるか事前確認
        kw_row = kw_scores.get(filename)
        if kw_row:
            any_candidate = any(
                float(kw_row.get(f"kw_score_{case}", 0.0) or 0.0)
                >= CASE_KW_THRESHOLD.get(case, RELEVANCE_THRESHOLD)
                for case in PREDEFINED_CASES
            )
            if not any_candidate:
                # 全事例で閾値未満 → LLMをスキップして None として記録
                rows.append({
                    "Filename":        filename,
                    "qwen_case_1":     "None",
                    "qwen_case_2":     "",
                    "qwen_raw":        "(skipped: all kw_scores below threshold)",
                    "kw_consistent_1": False,
                    "kw_consistent_2": None,
                })
                print(f"  [{i:3d}/{total}] ─ KW全閾値未満（LLMスキップ）: {filename[:55]}")
                continue

        # LLM 呼び出し
        prompt     = build_prompt(text, args.max_chars)
        raw        = call_ollama(prompt, args.model)
        cases_full = parse_response(raw)

        # KWスコアとの整合性チェック（事例別閾値を使用）
        consistent_flags = [
            kw_consistent(c, kw_row) for c in cases_full
        ]

        # 出力行の組み立て
        row = {
            "Filename":         filename,
            "qwen_case_1":      cases_full[0] if len(cases_full) > 0 else "None",
            "qwen_case_2":      cases_full[1] if len(cases_full) > 1 else "",
            "qwen_raw":         raw[:120],   # デバッグ用生レスポンス
            "kw_consistent_1":  consistent_flags[0] if consistent_flags else None,
            "kw_consistent_2":  consistent_flags[1] if len(consistent_flags) > 1 else None,
        }
        rows.append(row)

        dt  = time.time() - t0
        times.append(dt)
        avg = sum(times) / len(times)
        eta = avg * (total - i)

        case_str = " + ".join(cases_full) if cases_full else "None"
        cons_str = " / ".join(
            ("✓KW" if c else "✗KW") if c is not None else "（KW未確認）"
            for c in consistent_flags
        ) if consistent_flags else ""
        print(
            f"  [{i:3d}/{total}] {filename[:52]:<52}\n"
            f"    → {case_str}  {cons_str}"
            f"  {dt:.1f}s  ETA={fmt_eta(eta)}"
        )

    # CSV 書き出し
    fieldnames = [
        "Filename", "qwen_case_1", "qwen_case_2",
        "qwen_raw", "kw_consistent_1", "kw_consistent_2",
    ]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # サマリー
    elapsed = sum(times)
    print("\n" + "=" * 68)
    print(f"  完了: {len(rows)} 件  スキップ: {len(skipped)} 件")
    print(f"  処理時間: {fmt_eta(elapsed)}")
    print(f"  出力: {args.out}")

    if skipped:
        print(f"\n  ⚠️  テキストなし（要 pdf_extract.py 再実行）: {len(skipped)} 件")
        for fn in skipped[:5]:
            print(f"    python3 pdf_extract.py --files \"{fn}\" --force")
        if len(skipped) > 5:
            print(f"    ... 他 {len(skipped)-5} 件")

    # 事例別割当件数サマリー
    if rows:
        from collections import Counter
        case_count: Counter = Counter()
        for r in rows:
            for col in ("qwen_case_1", "qwen_case_2"):
                val = r.get(col, "")
                if val and val != "None":
                    case_count[val] += 1
        print("\n  【LLM 事例別割当件数】")
        for case in PREDEFINED_CASES:
            n = case_count.get(case, 0)
            print(f"  {n:3d}  {case}")

    print("=" * 68)


if __name__ == "__main__":
    main()