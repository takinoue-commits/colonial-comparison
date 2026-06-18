import os
import json
import pandas as pd
import requests
import re
import random
from pypdf import PdfReader

# ==========================================
# 1. あなたの環境に合わせてここをチェック
# ==========================================

# 分析したいPDFが入っているフォルダの場所
# 先ほど作った pdfs フォルダを指定しています
PDF_DIR = "./pdfs" 

# 結果を保存するファイル名
OUTPUT_CSV = "./analysis_results_local.csv"

# LLMのモデル名（手順1でダウンロードしたもの）
MODEL_NAME = "llama3.1"

# Ollamaが動いているURL（基本はこのまま）
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

# 分析の細かさ
# "paragraph"（段落単位）がローカルPCでは一番安定します
ANALYSIS_GRANULARITY = "paragraph" 

# ==========================================
# 2. 処理用の関数（中身は変えなくてOK）
# ==========================================

def extract_text_from_pdf(pdf_path):
    """PDFから文字を読み取る"""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        # 参考文献セクションなどは除外
        text = re.sub(r'References[\s\S]*$', '', text, flags=re.IGNORECASE)
        return text.strip()
    except Exception as e:
        print(f"エラー: {pdf_path} を読み込めませんでした。 {e}")
        return ""

def segment_text(text, granularity):
    """テキストを分割する"""
    if granularity == "paragraph":
        # 空行で区切って段落にする
        segs = re.split(r'\n\s*\n', text)
        # 短すぎる段落はゴミとして無視
        return [s.strip() for s in segs if len(s.split()) > 30]
    return [text]

def ask_local_llm(prompt):
    """あなたのPCのOllamaに質問を投げる"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json" # 返答をJSON形式に強制
    }
    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=300)
        if response.status_code == 200:
            return response.json().get("response")
        else:
            print(f"LLMエラー: {response.status_code}")
            return None
    except Exception as e:
        print(f"接続エラー: {e}")
        return None

# ==========================================
# 3. メインの実行部分
# ==========================================

def main():
    print(f"--- ローカルPDF分析を開始します ---")
    
    # PDFファイルを探す
    if not os.path.exists(PDF_DIR):
        print(f"エラー: {PDF_DIR} フォルダが見つかりません。")
        return
    
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    print(f"対象ファイル数: {len(pdf_files)}件")

    results = []

    for filename in pdf_files:
        print(f"\n■ {filename} を処理中...")
        pdf_path = os.path.join(PDF_DIR, filename)
        
        # 1. PDFから文字を出す
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text:
            continue
            
        # 2. 段落に分ける
        paragraphs = segment_text(full_text, ANALYSIS_GRANULARITY)
        
        # テストとして、1つのファイルから最大5段落だけランダムに選ぶ（時間はかかります）
        if len(paragraphs) > 5:
            paragraphs = random.sample(paragraphs, 5)

        for i, para in enumerate(paragraphs):
            print(f"  段落 {i+1}/{len(paragraphs)} を分析中...")
            
            prompt = f"""
            Analyze the following text for colonialism variables.
            Respond ONLY in JSON format with these keys:
            - case_id: name of the case
            - land_dispossession_score: 1-5
            - labor_exploitation_score: 1-5
            - cultural_assimilation_score: 1-5
            - evidence_quote: a short sentence from the text

            TEXT:
            {para}
            """
            
            answer = ask_local_llm(prompt)
            if answer:
                try:
                    data = json.loads(answer)
                    data["filename"] = filename
                    data["paragraph_index"] = i + 1
                    results.append(data)
                except:
                    print("  JSONの読み込みに失敗しました。")

        # ファイルごとにCSVへ保存（こまめに保存）
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print(f"\n--- すべて完了！ ---")
    print(f"結果は {OUTPUT_CSV} に保存されました。")

if __name__ == "__main__":
    main()