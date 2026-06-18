# -*- coding: utf-8 -*-
import pandas as pd
from scipy.stats import mannwhitneyu

INPUT_CSV = 'main_analysis_results.csv'
OUTPUT_DIFF_CSV = 'ainu_diff_vector.csv'
OUTPUT_STATS_CSV = 'ainu_statistical_tests.csv'

CASE_EDO = 'Ainu (Edo Period - Basho Ukeoi)'
CASE_MEIJI = 'Ainu (Meiji Period - Former Aborigine Law)'

DIMENSIONS = ['Land','Labor','Culture','Political','Economic']

def cliffs_delta(x, y):
    nx = len(x)
    ny = len(y)
    greater = 0
    less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                greater += 1
            elif xi < yj:
                less += 1
    return (greater - less) / (nx * ny)

def main():
    df = pd.read_csv(INPUT_CSV)
    df_edo = df[df['Case'] == CASE_EDO]
    df_meiji = df[df['Case'] == CASE_MEIJI]
    if df_edo.empty or df_meiji.empty:
        raise ValueError('Case labels not found in CSV')
    mean_edo = df_edo[DIMENSIONS].mean()
    mean_meiji = df_meiji[DIMENSIONS].mean()
    diff = mean_meiji - mean_edo
    diff_df = pd.DataFrame({
        'Edo_Mean': mean_edo,
        'Meiji_Mean': mean_meiji,
        'Difference_(Meiji_minus_Edo)': diff
    }).round(3)
    diff_df.to_csv(OUTPUT_DIFF_CSV)
    print('Difference vector saved:', OUTPUT_DIFF_CSV)
    rows = []
    for dim in DIMENSIONS:
        x = df_edo[dim]
        y = df_meiji[dim]
        u_stat, p_val = mannwhitneyu(x, y, alternative='two-sided')
        delta = cliffs_delta(x, y)
        rows.append({
            'Dimension': dim,
            'U_statistic': round(u_stat, 3),
            'p_value': round(p_val, 4),
            'Cliffs_Delta': round(delta, 3)
        })
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(OUTPUT_STATS_CSV, index=False)
    print('Stats table saved:', OUTPUT_STATS_CSV)

if __name__ == '__main__':
    main()
