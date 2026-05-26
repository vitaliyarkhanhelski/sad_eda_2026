"""
ANOVA i ANCOVA — Stage 4

Zakres:
- Jednoczynnikowa ANOVA (one-way): czy średnia zmiennej numerycznej różni się
  istotnie między grupami zmiennej kategorycznej?
- ANCOVA: czy efekt grupy pozostaje po kontroli covariatu?
  Przykład: czy MonthlyIncome różni się między JobRole po kontroli Age?

Zmienne wybrane na podstawie wyników Stage 2 (Mann-Whitney) i Stage 3 (Cramer's V):
  Numeryczne (dependent): MonthlyIncome, TotalWorkingYears, Age, JobLevel
  Kategoryczne (grouping): JobRole, Department, MaritalStatus, OverTime
  Covariate (ANCOVA): Age

Wejście: oczyszczony df z preprocessing.clean_data()
Wyjście: CSV z wynikami + wykresy boxplot wg grup
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from src.settings import CHARTS_STAGE4_DIR, DATA_STAGE4_DIR


# Zmienne do analizy — wybrane wg Mann-Whitney (p<0.05) i Cramer's V
NUMERIC_COLS = ['MonthlyIncome', 'TotalWorkingYears', 'Age', 'JobLevel']
CATEG_COLS   = ['JobRole', 'Department', 'MaritalStatus', 'OverTime']
COVARIATE    = 'Age'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Filtruje wiersze z brakami Attrition i koduje go binarnie."""
    df = df[df['Attrition'].notna()].copy()
    df['Attrition_bin'] = (df['Attrition'] == 'Yes').astype(int)
    return df


# ---------------------------------------------------------------------------
# 1. One-way ANOVA
# ---------------------------------------------------------------------------

def run_anova(df: pd.DataFrame) -> pd.DataFrame:
    """One-way ANOVA: czy średnia numeric_col różni się między grupami categ_col?

    Dla każdej pary (numeric, categorical) wykonuje test F (f_oneway).
    H0: wszystkie grupy mają tę samą średnią.
    p < 0.05 → odrzucamy H0 → grupy istotnie się różnią.
    """
    results = []
    for num_col in NUMERIC_COLS:
        for cat_col in CATEG_COLS:
            groups = []
            for g in df[cat_col].dropna().unique():
                group_values = df.loc[df[cat_col] == g, num_col].dropna()
                groups.append(group_values)
            f_stat, p_value = f_oneway(*groups)
            results.append({
                'zmienna_num': num_col,
                'zmienna_kat': cat_col,
                'F': round(f_stat, 3),
                'p_value': p_value,
                'istotny': p_value < 0.05,
            })

    results_df = pd.DataFrame(results).sort_values('p_value')
    DATA_STAGE4_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(DATA_STAGE4_DIR / 'anova_results.csv', index=False)
    print(f'[stage4] ANOVA — zapisano: anova_results.csv ({len(results_df)} par)')
    print(results_df.to_string(index=False))
    return results_df


def plot_anova_boxplots(df: pd.DataFrame, anova_results: pd.DataFrame) -> None:
    """Boxploty dla istotnych par (p < 0.05) z ANOVA — top 6 wg p-value."""
    significant = anova_results[anova_results['istotny']].head(6)
    if significant.empty:
        print('[stage4] Brak istotnych par ANOVA do wizualizacji.')
        return

    n = len(significant)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    axes = axes.flatten()

    palette = sns.color_palette('Set2')

    for i, (_, row) in enumerate(significant.iterrows()):
        num_col, cat_col = row['zmienna_num'], row['zmienna_kat']
        order = df.groupby(cat_col)[num_col].median().sort_values().index
        n_groups = df[cat_col].nunique()
        sns.boxplot(data=df, x=cat_col, y=num_col, order=order,
                    palette=palette[:n_groups], ax=axes[i])
        axes[i].set_title(f'{num_col} wg {cat_col}\n(F={row["F"]}, p={row["p_value"]:.2e})')
        axes[i].set_xlabel(cat_col)
        axes[i].set_ylabel(num_col)
        axes[i].tick_params(axis='x', rotation=70, labelsize=8)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('ANOVA — istotne różnice średnich między grupami', fontsize=14)
    plt.tight_layout()
    plt.savefig(CHARTS_STAGE4_DIR / 'anova_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[stage4] Zapisano: anova_boxplots.png')


# ---------------------------------------------------------------------------
# 2. ANCOVA
# ---------------------------------------------------------------------------

def run_ancova(df: pd.DataFrame) -> pd.DataFrame:
    """ANCOVA: czy efekt kategoryczny pozostaje po kontroli covariatu (Age)?

    Dla każdej pary (numeric, categorical) dopasowuje model OLS:
        numeric ~ C(categorical) + covariate
    i sprawdza p-value dla efektu kategorycznego po kontroli Age.
    """
    # ANCOVA nie ma sensu gdy covariate == dependent
    num_cols_ancova = [c for c in NUMERIC_COLS if c != COVARIATE]
    results = []

    for num_col in num_cols_ancova:
        for cat_col in CATEG_COLS:
            formula = f'{num_col} ~ C({cat_col}) + {COVARIATE}'
            model = ols(formula, data=df.dropna(subset=[num_col, cat_col, COVARIATE])).fit()
            anova_table = anova_lm(model, typ=2)

            p_cat = anova_table.loc[f'C({cat_col})', 'PR(>F)']
            p_cov = anova_table.loc[COVARIATE, 'PR(>F)']
            results.append({
                'zmienna_num': num_col,
                'zmienna_kat': cat_col,
                'covariate': COVARIATE,
                'p_categorical': round(p_cat, 4),
                'p_covariate': round(p_cov, 4),
                'kat_istotny': p_cat < 0.05,
                'cov_istotny': p_cov < 0.05,
            })

    results_df = pd.DataFrame(results).sort_values('p_categorical')
    results_df.to_csv(DATA_STAGE4_DIR / 'ancova_results.csv', index=False)
    print(f'[stage4] ANCOVA — zapisano: ancova_results.csv ({len(results_df)} par)')
    print(results_df.to_string(index=False))
    return results_df


# ---------------------------------------------------------------------------
# Punkt wejścia
# ---------------------------------------------------------------------------

def run(df: pd.DataFrame) -> None:
    """Uruchamia analizę ANOVA i ANCOVA."""
    CHARTS_STAGE4_DIR.mkdir(parents=True, exist_ok=True)
    DATA_STAGE4_DIR.mkdir(parents=True, exist_ok=True)

    df = _prepare(df)

    anova_results = run_anova(df)
    plot_anova_boxplots(df, anova_results)
    run_ancova(df)
