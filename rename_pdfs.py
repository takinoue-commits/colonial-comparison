# =========================================================
# rename_pdfs.py
# Rename PDFs based on extracted metadata
# Author +# Author + Year + Case
# =========================================================

import os
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "corpus"
CACHE_DIR = BASE_DIR / "extraction_cache"

# -----------------------------
# 表示用ケース名 → ファイル名用
# -----------------------------
CASE_SHORT = {
    "Ainu (Edo Period - Basho Ukeoi)": "Ainu-Edo",
    "Ainu (Meiji Period - Former Aborigine Law)": "Ainu-Meiji",
    "Aboriginal Australians": "Aboriginal-Australia",
    "Maori (New Zealand)": "Maori-NZ",
    "Native American (US)": "Native-US",
    "Taiwan (Japanese Rule)": "Taiwan-JapaneseRule",
    "Korea (Japanese Rule)": "Korea-JapaneseRule",
    "Indonesia (Dutch East Indies)": "Indonesia-Dutch",
    "Bengal (British India)": "Bengal-BritishIndia",
    "Ireland": "Ireland",
    "Ryukyu (Okinawa)": "Ryukyu-Okinawa",
}

# -----------------------------
# 文字列を安全化
# -----------------------------
def safe(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^\w\-_]", "", s)
    return s[:50] if len(s) > 50 else s

# -----------------------------
# 重複ファイル名を回避
# -----------------------------
def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    i = 2
    while True:
        new = path.with_name(f"{stem}_v{i}{suffix}")
        if not new.exists():
            return new
        i += 1

# -----------------------------
# main
# -----------------------------
def main():
    print("=" * 72)
    print(" rename_pdfs.py — rename PDFs using extracted metadata")
    print("=" * 72)

    if not PDF_DIR.exists():
        print("❌ corpus/ が見つかりません")
        return

    json_files = list(CACHE_DIR.glob("*.json"))
    if not json_files:
        print("❌ extraction_cache/ に JSON がありません")
        return

    for js in json_files:
        with open(js, encoding="utf-8") as f:
            data = json.load(f)

        orig_pdf = PDF_DIR / data.get("original_filename", "")
        if not orig_pdf.exists():
            print(f"⚠️ 元PDFが見つかりません: {orig_pdf.name}")
            continue

        author = safe(data.get("author", "UnknownAuthor"))
        year = data.get("year", "XXXX")

        case_guess = data.get("case_guess", "Unknown")
        case_part = CASE_SHORT.get(case_guess, "UnknownCase")

        new_name = f"{author}{year}_{case_part}.pdf"
        new_path = unique_path(PDF_DIR / new_name)

        orig_pdf.rename(new_path)
        print(f"✅ {orig_pdf.name} → {new_path.name}")

    print("\n✔ PDF renaming completed safely")

# -----------------------------
if __name__ == "__main__":
    main()
