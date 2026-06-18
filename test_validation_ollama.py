import os
import json
import pandas as pd
import requests
import re
import time
import random
import numpy as np
from pypdf import PdfReader

# ==========================================
# 1. テスト環境設定
# ==========================================
PDF_DIR = "./corpus_test"
OUTPUT_CSV = "./validation_test_twostep.csv"

# テスト対象を抽出するためのキーワード（制度名を追加）
TARGET_CASE_KEYWORDS = ["ainu", "hokkaido", "basho ukeoi", "kyu dojin", "former aborigine"]

SAMPLE_FILES_N = 5  
SAMPLE_PARAS_N = 10 

MODEL_NAME = "llama3.1" 
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

# 事例の分割定義
PREDEFINED_CASES = [
    "Ainu (Edo Period - Basho Ukeoi)", 
    "Ainu (Meiji Period - Former Aborigine Law)", 
    "Maori (New Zealand)", "Native American (US)", "Aboriginal Australians", 
    "Taiwan (Japanese Rule)", "Korea (Japanese Rule)", "Indonesia (Dutch East Indies)", 
    "Bengal (British India)", "Ireland", "Other", "Unknown"
]

# ==========================================
# 2. 処理用の関数
# ==========================================
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        text = re.sub(r'References[\s\S]*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Bibliography[\s\S]*$', '', text, flags=re.IGNORECASE)
        return text.strip()
    except Exception as e:
        print(f"  [エラー] 読み込み失敗 ({e})")
        return ""

def segment_text(text):
    raw_segs = re.split(r'\n\s*\n', text)
    valid_segs = []
    for seg in raw_segs:
        words = seg.strip().split()
        if len(words) < 30: 
            continue
        for i in range(0, len(words), 500):
            chunk = " ".join(words[i:i+500])
            valid_segs.append(chunk)
    return valid_segs

def ask_local_llm(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json" 
    }
    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=None)
        if response.status_code == 200:
            return response.json().get("response")
        else:
            print(f"  [LLMエラー] ステータスコード: {response.status_code}")
            return None
    except Exception as e:
        print(f"  [通信エラー] {e}")
        return None

# ==========================================
# 3. 二段階コーディング処理
# ==========================================
def extract_evidence_from_paragraph(paragraph):
    prompt = f"""
    You are an objective researcher. Read the text and extract ANY sentences that might be related to colonialism (land dispossession, labor exploitation, cultural assimilation, political control, economic extraction).
    If there is no relevant information, leave the quotes empty. DO NOT score them yet.

    OUTPUT EXACTLY THIS JSON STRUCTURE:
    {{
        "relevant_quotes": ["quote 1", "quote 2"]
    }}

    TEXT TO ANALYZE:
    {paragraph}
    """
    answer = ask_local_llm(prompt)
    if answer:
        try:
            data = json.loads(answer)
            return data.get("relevant_quotes", [])
        except:
            return []
    return []

def evaluate_document_holistically(all_quotes, cases_str):
    combined_quotes = "\n".join([f"- {q}" for q in all_quotes if q.strip()])
    
    if not combined_quotes.strip():
        return {
            "case_name": "Unknown", "period": "Unknown",
            "land_dispossession": {"score": 0.0, "reasoning": "No evidence extracted."},
            "labor_exploitation": {"score": 0.0, "reasoning": "No evidence extracted."},
            "cultural_assimilation": {"score": 0.0, "reasoning": "No evidence extracted."},
            "political_control": {"score": 0.0, "reasoning": "No evidence extracted."},
            "economic_extraction": {"score": 0.0, "reasoning": "No evidence extracted."}
        }

    prompt = f"""
    You are evaluating the overall intensity of colonialism in an academic paper based ONLY on the concentrated excerpts below.
    WARNING: These excerpts are concentrated negative signals. Do not automatically give high scores. Maintain a strict, objective baseline.
    
    STRICT RULES:
    a. Estimate 'case_name' from: [{cases_str}]. 
       - To carefully distinguish Ainu cases, look for period cues (Edo/Tokugawa period implies 'Basho Ukeoi', whereas Meiji/modern period implies 'Former Aborigine Law') and specific institutional terms like "place-name contract" vs. "former aborigine".
    b. Estimate 'period'.
    c. Evaluate the 5 negative colonialism variables on a scale of 0.0 to 5.0. 
       - BASELINE ASSUMPTION: Start at 0.0. You must justify any score above 0.0.
       - UNIVERSAL RUBRIC LIMITS:
         * 1.0-2.0: Isolated incidents, localized issues, or structurally minor disadvantages. Recognition of rights with some paternalism.
         * 3.0: Significant but partial exploitation/control. Mixed evidence, or policies affecting only a subset of the population.
         * 4.0: Severe, institutionalized, and widespread policies fundamentally altering indigenous life.
         * 5.0 (RESTRICTED): Reserved ONLY for absolute, totalizing destruction. e.g., Complete genocidal displacement, absolute slavery without exception, total eradication of language. DO NOT give 5.0 just because a harsh word is used in the text.
    d. Provide 'reasoning' explaining the overall score, directly referencing the severity and scope in the excerpts.

    OUTPUT EXACTLY THIS JSON STRUCTURE:
    {{
        "case_name": "...",
        "period": "...",
        "land_dispossession": {{"score": 0.0, "reasoning": "..."}},
        "labor_exploitation": {{"score": 0.0, "reasoning": "..."}},
        "cultural_assimilation": {{"score": 0.0, "reasoning": "..."}},
        "political_control": {{"score": 0.0, "reasoning": "..."}},
        "economic_extraction": {{"score": 0.0, "reasoning": "..."}}
    }}

    CONCENTRATED EXCERPTS FROM THE PAPER:
    {combined_quotes}
    """
    
    answer = ask_local_llm(prompt)
    if answer:
        try:
            return json.loads(answer)
        except:
            pass
    return None

def main():
    start_time = time.time()
    print(f"--- 二段階コーディングによる妥当性テスト (Ollama: {MODEL_NAME}) ---")
    
    if not os.path.exists(PDF_DIR):
        print(f"エラー: '{PDF_DIR}' が見つかりません。")
        return
    
    all_pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    if len(all_pdf_files) == 0: 
        print("エラー: フォルダ内にPDFファイルが存在しません。")
        return

    print(f"キーワード {TARGET_CASE_KEYWORDS} に一致するPDFを検索中...")
    target_files = []
    for filename in all_pdf_files:
        pdf_path = os.path.join(PDF_DIR, filename)
        try:
            text_preview = extract_text_from_pdf(pdf_path)[:10000].lower()
            if any(kw.lower() in text_preview for kw in TARGET_CASE_KEYWORDS):
                target_files.append(filename)
        except Exception as e:
            continue

    print(f"該当事例のPDFが {len(target_files)} 件見つかりました。")
    if len(target_files) == 0:
        print("テスト対象のPDFが見つかりませんでした。TARGET_CASE_KEYWORDS を変更してください。")
        return

    test_files = random.sample(target_files, min(SAMPLE_FILES_N, len(target_files)))
    print(f"その中から {len(test_files)} 件をランダムにサンプリングして分析します。\n")

    results = []
    cases_str = ", ".join(PREDEFINED_CASES)

    for file_idx, filename in enumerate(test_files):
        print(f"■ [{file_idx+1}/{len(test_files)}] {filename} を処理中...")
        pdf_path = os.path.join(PDF_DIR, filename)
        
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text: continue
            
        paragraphs = segment_text(full_text)
        test_paragraphs = random.sample(paragraphs, min(SAMPLE_PARAS_N, len(paragraphs)))
        
        all_collected_quotes = []
        
        for i, para in enumerate(test_paragraphs):
            print(f"  [Step 1] 段落 {i+1}/{len(test_paragraphs)} から証拠を抽出...", end=" ", flush=True)
            quotes = extract_evidence_from_paragraph(para)
            if quotes:
                all_collected_quotes.extend(quotes)
                print(f"✅ {len(quotes)}件抽出")
            else:
                print("なし")
                
        print(f"  [Step 2] 収集した全証拠（{len(all_collected_quotes)}件）を基に論文全体を総合評価中...", end=" ", flush=True)
        final_eval = evaluate_document_holistically(all_collected_quotes, cases_str)
        
        if final_eval:
            row = {
                "Filename": filename,
                "Extracted_Quotes_Count": len(all_collected_quotes),
                "Case_Name": final_eval.get("case_name", "Unknown"),
                "Period": final_eval.get("period", "Unknown"),
                "Overall_Land_Score": final_eval.get("land_dispossession", {}).get("score", 0.0),
                "Land_Reasoning": final_eval.get("land_dispossession", {}).get("reasoning", ""),
                "Overall_Labor_Score": final_eval.get("labor_exploitation", {}).get("score", 0.0),
                "Labor_Reasoning": final_eval.get("labor_exploitation", {}).get("reasoning", ""),
                "Overall_Culture_Score": final_eval.get("cultural_assimilation", {}).get("score", 0.0),
                "Culture_Reasoning": final_eval.get("cultural_assimilation", {}).get("reasoning", ""),
                "Overall_Political_Score": final_eval.get("political_control", {}).get("score", 0.0),
                "Political_Reasoning": final_eval.get("political_control", {}).get("reasoning", ""),
                "Overall_Economic_Score": final_eval.get("economic_extraction", {}).get("score", 0.0),
                "Economic_Reasoning": final_eval.get("economic_extraction", {}).get("reasoning", "")
            }
            results.append(row)
            print("✅ 完了")
        else:
            print("❌ 失敗")

        if results:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    elapsed = time.time() - start_time
    m, s = divmod(elapsed, 60)
    print(f"\n--- テスト完了 (所要時間: {int(m)}分 {int(s)}秒) ---")
    print(f"テスト結果を {OUTPUT_CSV} に保存しました。\n")

    if results:
        print("=== テスト対象事例の点数傾向（一貫性の確認） ===")
        df_summary = pd.DataFrame(results)
        score_cols = ["Overall_Land_Score", "Overall_Labor_Score", "Overall_Culture_Score", "Overall_Political_Score", "Overall_Economic_Score"]
        
        for col in score_cols:
            df_summary[col] = pd.to_numeric(df_summary[col], errors='coerce').fillna(0.0)
            
        print(f"分析したファイル数: {len(df_summary)} 件")
        for col in score_cols:
            mean_val = df_summary[col].mean()
            std_val = df_summary[col].std()
            min_val = df_summary[col].min()
            max_val = df_summary[col].max()
            var_name = col.replace("Overall_", "").replace("_Score", "")
            print(f"  - {var_name:10s} : 平均 {mean_val:.1f} (標準偏差: {std_val:.2f}, 最小 {min_val:.1f} ~ 最大 {max_val:.1f})")
            
        print("\n※ 評価ガイド:")
        print("- 標準偏差（データのばらつき）が小さいほど、LLMの評価基準が安定していると解釈できます。")
        print("- Zスコア等の標準化処理を行うことで、変数のスケールを揃え、事例間の相対的な強度比較が可能になります。")

if __name__ == "__main__":
    main()