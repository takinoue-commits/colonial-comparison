"""
statistics.py
FINAL statistics summary for Ainu (Edo vs Meiji)

Reads already-valid results from:
- analysis_ainu_factor_tests.csv

No re-computation. No fragile assumptions.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "analysis_ainu_factor_tests.csv")

def main():
    print("=" * 80)
    print(" statistics.py — Ainu Edo vs Meiji (FINAL, AUTHORITATIVE)")
    print("=" * 80)

    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(f"❌ Missing: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    print("\n### Mann–Whitney U tests + Cliff’s Delta (paper-level) ###\n")
    print(df.to_string(index=False))

    print("\n✅ statistics.py completed successfully")
    print("=" * 80)

if __name__ == "__main__":
    main()