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
OUTPUT_CSV = "./validation_test_twostep_iterative.csv"

# テスト対象を抽出するためのキーワード
TARGET_CASE_KEYWORDS = ["native american", "indian removal", "reservation", "boarding school", "indigenous peoples in the us"]

# サンプリング設定
SAMPLE_FILES_N = 5  
SAMPLE_PARAS_N = 10 

# 【統計的検証の追加】反復シミュレーションの回数 (n=30)
ITERATION_N = 30

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
            return None
    except Exception:
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
            "land_dispossession": {"score": None, "reasoning": "No evidence extracted."},
            "labor_exploitation": {"score": None, "reasoning": "No evidence extracted."},
            "cultural_assimilation": {"score": None, "reasoning": "No evidence extracted."},
            "political_control": {"score": None, "reasoning": "No evidence extracted."},
            "economic_extraction": {"score": None, "reasoning": "No evidence extracted."}
        }

    prompt = f"""
    You are evaluating the overall intensity of colonialism in an academic paper based ONLY on the concentrated excerpts below.
    
    STRICT RULES:
    a. Estimate 'case_name' from: [{cases_str}]. 
    b. Estimate 'period'.
    c. Evaluate the 5 negative colonialism variables on a scale of 1.0 to 5.0, or use `null` if completely unmentioned.
       - IMPORTANT DISTINCTION (null vs 1.0): 
         * If the text says absolutely NOTHING about the variable, you MUST set the score to `null` (not 0.0, but null).
         * If the text mentions the variable but describes a fair, equitable, or non-exploitative situation, score it 1.0 or 2.0.
       - UNIVERSAL RUBRIC LIMITS:
         * 1.0-2.0: Isolated incidents, localized issues, or structurally minor disadvantages. Recognition of rights with some paternalism.
         * 3.0: Significant but partial exploitation/control. Mixed evidence, or policies affecting only a subset of the population.
         * 4.0: Severe, institutionalized, and widespread policies fundamentally altering indigenous life.
         * 5.0 (RESTRICTED): Reserved ONLY for absolute, totalizing destruction. e.g., Complete genocidal displacement, absolute slavery without exception, total eradication of language.
    d. Provide 'reasoning' explaining the score. If `null`, explain that evidence is completely absent.

    OUTPUT EXACTLY THIS JSON STRUCTURE:
    {{
        "case_name": "...",
        "period": "...",
        "land_dispossession": {{"score": null, "reasoning": "..."}},
        "labor_exploitation": {{"score": null, "reasoning": "..."}},
        "cultural_assimilation": {{"score": null, "reasoning": "..."}},
        "political_control": {{"score": null, "reasoning": "..."}},
        "economic_extraction": {{"score": null, "reasoning": "..."}}
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
    total_start_time = time.time()
    print(f"--- 統計的妥当性検証シミュレーション (Ollama: {MODEL_NAME}, 反復回数: {ITERATION_N}) ---")
    
    if not os.path.exists(PDF_DIR):
        print(f"エラー: '{PDF_DIR}' が見つかりません。")
        return
    
    all_pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    if len(all_pdf_files) == 0: 
        print("エラー: フォルダ内にPDFファイルが存在しません。")
        return

    print(f"キーワード {TARGET_CASE_KEYWORDS} に一致するPDFを検索中...")
    target_files = []
    # 処理の高速化のため、テキスト抽出済みデータをキャッシュする
    text_cache = {}
    
    for filename in all_pdf_files:
        pdf_path = os.path.join(PDF_DIR, filename)
        try:
            full_text = extract_text_from_pdf(pdf_path)
            if full_text:
                text_preview = full_text[:10000].lower()
                if any(kw.lower() in text_preview for kw in TARGET_CASE_KEYWORDS):
                    target_files.append(filename)
                    text_cache[filename] = full_text
        except Exception:
            continue

    print(f"該当事例のPDFが {len(target_files)} 件見つかりました。\n")
    if len(target_files) == 0:
        print("テスト対象のPDFが見つかりませんでした。")
        return

    all_results = []
    cases_str = ", ".join(PREDEFINED_CASES)

    # モンテカルロ法的な反復シミュレーションの実行
    for iteration in range(ITERATION_N):
        print(f"=== イテレーション [{iteration + 1}/{ITERATION_N}] ===")
        
        # 毎回異なる組み合わせをランダムにサンプリング（抽出ファイルの偏りを防ぐ）
        test_files = random.sample(target_files, min(SAMPLE_FILES_N, len(target_files)))
        
        for file_idx, filename in enumerate(test_files):
            print(f"  ■ ファイル [{file_idx+1}/{len(test_files)}] {filename} を処理中...")
            
            full_text = text_cache[filename]
            paragraphs = segment_text(full_text)
            
            # 段落もイテレーションごとにランダムにサンプリング（テキストの局所的依存性を平滑化）
            test_paragraphs = random.sample(paragraphs, min(SAMPLE_PARAS_N, len(paragraphs)))
            all_collected_quotes = []
            
            for para in test_paragraphs:
                quotes = extract_evidence_from_paragraph(para)
                if quotes:
                    all_collected_quotes.extend(quotes)
                    
            final_eval = evaluate_document_holistically(all_collected_quotes, cases_str)
            
            if final_eval:
                row = {
                    "Iteration": iteration + 1,
                    "Filename": filename,
                    "Extracted_Quotes_Count": len(all_collected_quotes),
                    "Case_Name": final_eval.get("case_name", "Unknown"),
                    "Period": final_eval.get("period", "Unknown"),
                    "Overall_Land_Score": final_eval.get("land_dispossession", {}).get("score"),
                    "Overall_Labor_Score": final_eval.get("labor_exploitation", {}).get("score"),
                    "Overall_Culture_Score": final_eval.get("cultural_assimilation", {}).get("score"),
                    "Overall_Political_Score": final_eval.get("political_control", {}).get("score"),
                    "Overall_Economic_Score": final_eval.get("economic_extraction", {}).get("score")
                }
                all_results.append(row)
                print("    ✅ 評価完了")
            else:
                print("    ❌ 評価失敗")

        # 進捗を随時保存
        if all_results:
            pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    elapsed = time.time() - total_start_time
    m, s = divmod(elapsed, 60)
    print(f"\n--- 全シミュレーション完了 (総所要時間: {int(m)}分 {int(s)}秒) ---")

    # 統計的安定性の分析
    if all_results:
        print("\n=== 統計的安定性の検証結果（全イテレーション集計） ===")
        df_summary = pd.DataFrame(all_results)
        score_cols = ["Overall_Land_Score", "Overall_Labor_Score", "Overall_Culture_Score", "Overall_Political_Score", "Overall_Economic_Score"]
        
        for col in score_cols:
            df_summary[col] = pd.to_numeric(df_summary[col], errors='coerce')
            
        print(f"総処理サンプル数: {len(df_summary)} 件")
        
        for col in score_cols:
            # 欠損値（null）を除外して真の強度を計算
            valid_data = df_summary[col].dropna()
            
            if valid_data.empty:
                print(f"  - {col.replace('Overall_', '').replace('_Score', ''):10s} : データなし")
            else:
                mean_val = valid_data.mean()
                std_val = valid_data.std() if len(valid_data) > 1 else 0.0
                se_val = std_val / np.sqrt(len(valid_data)) # 標準誤差（Standard Error）を計算
                
                print(f"  - {col.replace('Overall_', '').replace('_Score', ''):10s} : "
                      f"平均 {mean_val:.2f} (標準誤差: {se_val:.3f}, 標準偏差: {std_val:.2f})")
                
        print("\n※ 学術的解釈ガイド:")
        print("- 標準誤差（SE）が小さい（例: 0.2未満）ほど、母平均の推定が統計的に安定しており、LLMの測定の信頼性が高いことを示します。")
        print("- 反復処理によって極端な外れ値が平滑化され、特定の変数が持つ構造的な強度がより客観的に可視化されています。")

if __name__ == "__main__":
    main()