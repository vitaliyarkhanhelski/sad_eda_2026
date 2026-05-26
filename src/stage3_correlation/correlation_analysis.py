"""
Analiza korelacji i wizualizacje — Stage 3

Zakres:
- Macierz korelacji Spearmana dla zmiennych numerycznych:
    * heatmapa
    * top 10 zmiennych najbardziej skorelowanych z Attrition
- Korelacja między zmiennymi kategorycznymi:
    * Cramer's V — heatmapa asocjacji
- Scatter matrix (pair plot) dla top zmiennych numerycznych,
  z kolorowaniem wg Attrition
- Parallel Coordinates — wizualizacja profili pracowników odchodzących vs pozostających
- Analiza Attrition rate wg grup:
    * wg Department, JobRole, MaritalStatus, OverTime, BusinessTravel
    * wykresy słupkowe z % odejść

Wejście: oczyszczony df z preprocessing.clean_data()
Wyjście: wykresy PNG + CSV z top korelacjami
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler

from src.settings import CHARTS_STAGE3_DIR, DATA_STAGE3_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Filtruje wiersze z brakami Attrition i koduje go binarnie."""
    df = df[df['Attrition'].notna()].copy()
    df['Attrition_bin'] = (df['Attrition'] == 'Yes').astype(int)
    return df


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include='number').columns.tolist()



def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V — miara siły związku między dwiema zmiennymi kategorycznymi (zakres 0–1).

    1. Tworzy tablicę krzyżową (ile razy każda para kategorii wystąpiła razem).
    2. Liczy test chi-kwadrat — sprawdza czy rozkład różni się od niezależnego.
    3. Normalizuje wynik chi2 do przedziału 0–1 wzorem: sqrt(chi2 / (n * k))
       gdzie n = liczba obserwacji, k = min(wiersze, kolumny) - 1.
    Wynik 0 = brak związku, 1 = pełny związek.
    """
    confusion = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion)
    n = confusion.sum().sum()
    k = min(confusion.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


# ---------------------------------------------------------------------------
# 1. Macierz korelacji Spearmana
# ---------------------------------------------------------------------------

def plot_spearman_heatmap(df: pd.DataFrame) -> None:
    """Heatmapa korelacji Spearmana dla zmiennych numerycznych (razem z Attrition_bin)."""
    num_cols = _numeric_cols(df)
    corr = df[num_cols].corr(method='spearman')

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
        center=0, linewidths=0.4, annot_kws={'size': 7},
        vmin=-1, vmax=1, ax=ax,
    )
    ax.set_title('Macierz korelacji Spearmana (zmienne numeryczne)', fontsize=13)
    plt.tight_layout()
    plt.savefig(CHARTS_STAGE3_DIR / 'spearman_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[stage3] Zapisano: spearman_heatmap.png')


def save_top_correlations(df: pd.DataFrame, top_n: int = 10) -> None:
    """Zapisuje top N zmiennych najsilniej skorelowanych z Attrition (Spearman)."""
    num_cols = _numeric_cols(df)
    corr = df[num_cols].corr(method='spearman')

    top = (
        corr['Attrition_bin']
        .drop('Attrition_bin')
        .abs()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    top.columns = ['kolumna', 'korelacja_abs']
    top['korelacja'] = corr.loc[top['kolumna'], 'Attrition_bin'].values

    DATA_STAGE3_DIR.mkdir(parents=True, exist_ok=True)
    top.to_csv(DATA_STAGE3_DIR / 'top_correlations.csv', index=False)
    print(f'[stage3] Zapisano: top_correlations.csv ({top_n} kolumn)')
    print(top.to_string(index=False))


# ---------------------------------------------------------------------------
# 2. Cramer's V — heatmapa asocjacji kategorycznych
# ---------------------------------------------------------------------------

def plot_cramers_v_heatmap(df: pd.DataFrame) -> None:
    """Heatmapa siły związku (Cramer's V) między zmiennymi kategorycznymi."""
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    v_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)

    for col_a in cat_cols:
        for col_b in cat_cols:
            v_matrix.loc[col_a, col_b] = _cramers_v(df[col_a], df[col_b])

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(v_matrix, dtype=bool))
    sns.heatmap(
        v_matrix.astype(float), mask=mask, annot=True, fmt='.2f',
        cmap='YlOrRd', vmin=0, vmax=1, linewidths=0.4,
        annot_kws={'size': 8}, ax=ax,
    )
    ax.set_title("Cramer's V — siła związku między zmiennymi kategorycznymi", fontsize=13)
    plt.tight_layout()
    plt.savefig(CHARTS_STAGE3_DIR / 'cramers_v_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[stage3] Zapisano: cramers_v_heatmap.png")


# ---------------------------------------------------------------------------
# 3. Attrition rate wg grup
# ---------------------------------------------------------------------------

def plot_attrition_rate_by_groups(df: pd.DataFrame, top_n: int = 6) -> None:
    """Wykresy słupkowe % odejść wg top N zmiennych kategorycznych (wg Cramer's V z Attrition)."""
    cat_cols = df.select_dtypes(include='object').columns.difference(['Attrition']).tolist()
    group_cols = sorted(cat_cols, key=lambda c: _cramers_v(df[c], df['Attrition']), reverse=True)[:top_n]
    print(f'[stage3] Attrition rate — top {top_n} kategorycznych (Cramer\'s V): {group_cols}')

    n_cols = 3
    n_rows = math.ceil(top_n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(group_cols):
        rates = (
            df.groupby(col)['Attrition_bin']
            .mean()
            .mul(100)
            .sort_values(ascending=True)
        )
        bars = axes[i].barh(rates.index, rates.values, color='#e05c5c')
        axes[i].set_xlabel('Attrition rate (%)')
        axes[i].set_title(f'Attrition rate wg {col}')
        axes[i].set_xlim(0, rates.max() * 1.15)

        for bar, val in zip(bars, rates.values):
            axes[i].text(
                val + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=9,
            )

    for j in range(len(group_cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Attrition rate wg grup demograficznych i zawodowych', fontsize=14)
    plt.tight_layout()
    plt.savefig(CHARTS_STAGE3_DIR / 'attrition_rate_by_groups.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[stage3] Zapisano: attrition_rate_by_groups.png')


# ---------------------------------------------------------------------------
# 4. Parallel Coordinates
# ---------------------------------------------------------------------------

def plot_parallel_coordinates(df: pd.DataFrame, top_n: int = 6) -> None:
    """Parallel coordinates — profile pracowników odchodzących vs pozostających."""
    num_cols = _numeric_cols(df)
    corr_with_attrition = (
        df[num_cols].corr(method='spearman')['Attrition_bin']
        .drop('Attrition_bin')
        .abs()
        .sort_values(ascending=False)
    )
    top_cols = corr_with_attrition.head(top_n).index.tolist()

    # Normalizacja do [0, 1] żeby osie były porównywalne
    subset = df[top_cols + ['Attrition']].copy()
    subset[top_cols] = MinMaxScaler().fit_transform(subset[top_cols])

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {'Yes': '#e05c5c', 'No': '#4c8cbf'}
    alphas = {'Yes': 0.4, 'No': 0.1}

    for attrition_val in ['No', 'Yes']:
        group = subset[subset['Attrition'] == attrition_val][top_cols]
        for _, row in group.iterrows():
            ax.plot(top_cols, row.values, color=colors[attrition_val],
                    alpha=alphas[attrition_val], linewidth=0.6)

    # Linie legendy
    for val, color in colors.items():
        ax.plot([], [], color=color, linewidth=2, label=f'Attrition={val}')

    ax.set_xticks(range(len(top_cols)))
    ax.set_xticklabels(top_cols, rotation=20, ha='right')
    ax.set_ylabel('Wartość znormalizowana [0–1]')
    ax.set_title(f'Parallel Coordinates — profile pracowników (top {top_n} zmiennych)', fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_STAGE3_DIR / 'parallel_coordinates.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[stage3] Zapisano: parallel_coordinates.png')


# ---------------------------------------------------------------------------
# Punkt wejścia
# ---------------------------------------------------------------------------

def run(df: pd.DataFrame) -> None:
    """Uruchamia analizę korelacji i generuje wizualizacje."""
    CHARTS_STAGE3_DIR.mkdir(parents=True, exist_ok=True)
    DATA_STAGE3_DIR.mkdir(parents=True, exist_ok=True)

    df = _prepare(df)

    plot_spearman_heatmap(df)
    save_top_correlations(df)
    plot_cramers_v_heatmap(df)
    plot_attrition_rate_by_groups(df)
    plot_parallel_coordinates(df)
