# =========================================================
# pdf_extract.py
# Phase 1: PDF full-text extraction + metadata caching
#
# 【prescreen.py との連携仕様】
#   prescreen.py は以下のファイルをキャッシュとして参照する:
#     extraction_cache/<filename>.pdf.txt   ← 本文テキスト
#     extraction_cache/<filename>.pdf.meta  ← {"page_count": N} の JSON
#
# 【実行例】
#   python3 pdf_extract.py               # pdfs/ 内の全PDFを処理
#   python3 pdf_extract.py --force       # キャッシュ済みも再処理
#   python3 pdf_extract.py --files a.pdf b.pdf --force
# =========================================================

import re
import json
import argparse
import sys
import warnings
from typing import DefaultDict, Dict, List, Optional, Set, Tuple
from pathlib import Path

# pdfminer: テキスト抽出
from pdfminer.high_level import extract_text as pdfminer_extract
from pdfminer.pdfpage import PDFPage

# pypdf: ページ数取得・メタデータ取得（PyPDF2の後継）
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader  # フォールバック（旧環境）

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Paths  ※ prescreen.py の CORPUS_DIR / CACHE_DIR と一致させる
# ---------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parent
PDF_DIR   = BASE_DIR / "corpus"          # prescreen.py: CORPUS_DIR = "./pdfs"
CACHE_DIR = BASE_DIR / "extraction_cache"
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# 事例別キーワード（スクリーニング補助用）
# ※ prescreen.py の CASE_KEYWORDS と重複するが、
#   抽出側でも事例推定ログを残すために保持する
# ---------------------------------------------------------
CASE_KEYWORDS: Dict[str, List[str]] = {
    "Ainu (Edo Period - Basho Ukeoi)": [
        "ainu", "アイヌ", "ezo", "ezochi", "蝦夷", "蝦夷地",
        "basho", "basho ukeoi", "場所請負", "tokugawa", "edo", "matsumae",
    ],
    "Ainu (Meiji Period - Former Aborigine Law)": [
        "ainu", "アイヌ", "former aborigine", "旧土人",
        "assimilation", "同化", "hokkaido colonization", "北海道開拓", "meiji",
    ],
    "Aboriginal Australians": [
        "aboriginal", "aborigines", "torres strait islander",
        "assimilation", "stolen generations",
    ],
    "Maori (New Zealand)": [
        "maori", "māori", "waitangi", "aotearoa",
    ],
    "Native American (US)": [
        "native american", "american indian",
        "tribal", "reservation", "indigenous peoples of the united states",
    ],
    "Taiwan (Japanese Rule)": [
        "taiwan", "台湾", "japanese colonial", "日本統治",
        "governor-general of taiwan", "kominka",
    ],
    "Korea (Japanese Rule)": [
        "korea", "朝鮮", "annexation of korea", "韓国併合", "land survey project",
    ],
    "Indonesia (Dutch East Indies)": [
        "dutch east indies", "indonesia", "indonesian",
        "java", "javanese", "cultivation system",
    ],
    "Bengal (British India)": [
        "bengal", "bengali", "east india company", "permanent settlement",
    ],
    "Ireland": [
        "ireland", "irish", "great famine", "colonial ireland",
    ],
    "Ryukyu (Okinawa)": [
        "ryukyu", "琉球", "okinawa", "沖縄",
        "ryukyu disposition", "琉球処分",
    ],
}

# ---------------------------------------------------------
# テキスト分割（head / middle / tail の代表ゾーン）
# ---------------------------------------------------------
def split_zones(text: str) -> List[str]:
    """長文を先頭・中央・末尾に分割してキーワード検索を効率化する。"""
    n = len(text)
    if n <= 6000:
        return [text]
    mid = n // 2
    return [
        text[:3000],
        text[mid - 1500 : mid + 1500],
        text[-3000:],
    ]


# ---------------------------------------------------------
# メタデータ補助
# ---------------------------------------------------------
def extract_year(text: str) -> str:
    """テキスト先頭から発行年を推定する。"""
    m = re.search(r"\b(18|19|20)\d{2}\b", text[:5000])
    return m.group(0) if m else "XXXX"


def extract_author(meta_author: str, text_head: str) -> str:
    """PDFメタデータ優先で第一著者を推定する。"""
    if meta_author and meta_author.strip():
        # "Last, First" や "First Last" から姓部分を取得
        return meta_author.split(",")[0].strip()
    # テキスト先頭から大文字始まりの単語を推定
    m = re.search(r"\b([A-Z][A-Za-z\-]{2,})\b", text_head[:1000])
    return m.group(1) if m else "UnknownAuthor"


# ---------------------------------------------------------
# 事例キーワードヒット数スコアリング
# ※ 旧版は「ヒットあり/なし」だけだったが、
#   ヒット数の多い事例を優先するよう改善
# ---------------------------------------------------------
def score_case_keywords(text: str) -> Dict[str, int]:
    """各事例のキーワードヒット数を返す（0は除外）。"""
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    for case, kws in CASE_KEYWORDS.items():
        count = sum(1 for kw in kws if kw.lower() in text_lower)
        if count > 0:
            scores[case] = count
    return scores


def best_case_guess(scores: Dict[str, int]) -> str:
    """ヒット数が最多の事例を返す。同点の場合はアルファベット順で安定化。"""
    if not scores:
        return "Unknown"
    return max(scores, key=lambda c: (scores[c], c))


# ---------------------------------------------------------
# pypdf でのメタデータ取得（互換ラッパー）
# ・pypdf  : reader.metadata.get("/Author")
# ・PyPDF2 : reader.metadata.author  (非推奨の旧 API)
# ---------------------------------------------------------
def get_pdf_metadata(reader) -> Tuple[str, int]:
    """(author_str, page_count) を返す。失敗時は ("", 0)。"""
    page_count = len(reader.pages)
    try:
        meta = reader.metadata
        # pypdf は dict ライクなので /Author キーで取得
        author = (meta.get("/Author") or meta.get("author") or "") if meta else ""
        # PyPDF2 旧 API へのフォールバック
        if not author and hasattr(meta, "author"):
            author = meta.author or ""
    except Exception:
        author = ""
    return str(author).strip(), page_count


# ---------------------------------------------------------
# PDF テキスト抽出（pdfminer → pypdf フォールバック）
# ---------------------------------------------------------
def extract_pdf_text(pdf_path: Path) -> Tuple[str, str]:
    """
    (text, method) を返す。
    スキャン PDF（テキスト層なし）は空文字が返るため、
    OCR が必要な場合は別途 pdf_extract_ocr.py を使用すること。
    """
    # --- pdfminer（主要手段）---
    try:
        text = pdfminer_extract(str(pdf_path)) or ""
        if text.strip():
            return _clean_text(text), "pdfminer"
    except Exception as e:
        print(f"    [警告] pdfminer 失敗: {e}")

    # --- pypdf フォールバック ---
    try:
        reader = PdfReader(str(pdf_path))
        pages  = [p.extract_text() or "" for p in reader.pages]
        text   = "\n\n".join(pages)
        if text.strip():
            return _clean_text(text), "pypdf"
    except Exception as e:
        print(f"    [警告] pypdf 失敗: {e}")

    return "", "none"


def _clean_text(text: str) -> str:
    """参考文献以降を除去して本文のみを残す。"""
    text = re.sub(r"\bReferences\b[\s\S]*$",   "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bBibliography\b[\s\S]*$", "", text, flags=re.IGNORECASE)
    return text.strip()


# ---------------------------------------------------------
# キャッシュ書き込み（prescreen.py 互換フォーマット）
# ---------------------------------------------------------
def write_cache(pdf_path: Path, text: str, page_count: int) -> None:
    """
    prescreen.py が期待するキャッシュを書き出す:
      <filename>.pdf.txt   ← 抽出テキスト
      <filename>.pdf.meta  ← {"page_count": N}
    """
    stem = pdf_path.name          # "foo.pdf"（拡張子込み）
    txt_path  = CACHE_DIR / f"{stem}.txt"
    meta_path = CACHE_DIR / f"{stem}.meta"

    txt_path.write_text(text, encoding="utf-8")
    meta_path.write_text(
        json.dumps({"page_count": page_count}, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------
def process_pdf(pdf_path: Path, force: bool) -> dict:
    """1 件の PDF を処理してキャッシュを書き出す。結果サマリーを返す。"""
    stem = pdf_path.name
    txt_path = CACHE_DIR / f"{stem}.txt"

    # キャッシュ済みスキップ
    if txt_path.exists() and not force:
        print(f"  [スキップ] {pdf_path.name}（--force で再処理）")
        return {"file": pdf_path.name, "status": "skipped"}

    print(f"  処理中: {pdf_path.name}")

    # テキスト抽出
    text, method = extract_pdf_text(pdf_path)

    # ページ数・著者メタデータ
    page_count = 0
    author     = "UnknownAuthor"
    try:
        reader             = PdfReader(str(pdf_path))
        author_raw, page_count = get_pdf_metadata(reader)
        author             = extract_author(author_raw, text[:2000])
    except Exception as e:
        print(f"    [警告] メタデータ取得失敗: {e}")

    year       = extract_year(text)
    kw_scores  = score_case_keywords(text)
    case_guess = best_case_guess(kw_scores)
    word_count = len(re.findall(r"[a-zA-Z]{2,}|[\u3040-\u9FFF]{2,}", text))

    # キャッシュ書き込み
    if text:
        write_cache(pdf_path, text, page_count)
        status = f"OK [{method}]  {word_count:,}語  {page_count}頁  → {case_guess}"
    else:
        status = "テキスト抽出失敗（スキャンPDFの可能性。OCR処理が必要）"
        # 空テキストでもキャッシュは書いておく（再試行マーカー）
        write_cache(pdf_path, "", page_count)

    print(f"    {status}")

    return {
        "file":       pdf_path.name,
        "status":     status,
        "author":     author,
        "year":       year,
        "case_guess": case_guess,
        "kw_scores":  kw_scores,
        "word_count": word_count,
        "page_count": page_count,
        "method":     method,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF テキスト抽出 → extraction_cache/ に保存"
    )
    parser.add_argument(
        "--files", nargs="+", metavar="FILENAME",
        help="処理対象 PDF ファイル名を指定（省略時は pdfs/ 内の全 PDF）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="キャッシュ済みでも再処理する",
    )
    args = parser.parse_args()

    if not PDF_DIR.exists():
        print(f"エラー: '{PDF_DIR}' が見つかりません。")
        sys.exit(1)

    # 処理対象ファイルの決定
    if args.files:
        pdf_files = []
        for name in args.files:
            p = PDF_DIR / name
            if p.exists():
                pdf_files.append(p)
            else:
                print(f"  [警告] 見つかりません: {p}")
    else:
        pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("処理対象の PDF がありません。")
        sys.exit(0)

    print("=" * 68)
    print(f"  pdf_extract.py  対象: {len(pdf_files)} 件  force={args.force}")
    print(f"  corpus : {PDF_DIR}")
    print(f"  cache  : {CACHE_DIR}")
    print("=" * 68)

    results  = []
    failed   = []
    skipped  = 0

    for pdf_path in pdf_files:
        result = process_pdf(pdf_path, force=args.force)
        results.append(result)
        if result["status"] == "skipped":
            skipped += 1
        elif "失敗" in result["status"]:
            failed.append(pdf_path.name)

    # サマリー
    print("\n" + "=" * 68)
    print(f"  完了: {len(results)} 件  スキップ: {skipped}  抽出失敗: {len(failed)}")
    if failed:
        print(f"\n  ⚠️  OCR が必要なファイル（{len(failed)} 件）:")
        for fn in failed:
            print(f"    python3 pdf_extract.py --files \"{fn}\" --force")
    print("=" * 68)


if __name__ == "__main__":
    main()