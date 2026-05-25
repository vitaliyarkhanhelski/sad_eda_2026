"""
Analiza opisowa i wnioskowanie statystyczne — Etap 2

Zakres:
- Analiza zmiennej docelowej Attrition (rozkład, niezbalansowanie klas)
- Statystyki opisowe zmiennych numerycznych (mean, std, skewness, kurtosis, CV)
- Histogramy i boxploty z podziałem na Attrition=Yes/No
- Testy statystyczne:
    * zmienne numeryczne vs Attrition → test Manna-Whitneya (nieparametryczny)
    * zmienne kategoryczne vs Attrition → test Chi-kwadrat
- Ranking istotności zmiennych wg p-value

Wejście: oczyszczony df z preprocessing.clean_data()
Wyjście: wykresy (charts/stage2_descriptive/) + CSV (data/stage2_descriptive/)

Uwaga: do testów używamy tylko wierszy z wypełnionym Attrition (1320 wierszy).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency

from src.settings import CHARTS_STAGE2_DIR, DATA_STAGE2_DIR


def run(df: pd.DataFrame) -> None:
    """Uruchamia pełną analizę opisową i statystyczną."""
    print("\n[stage2] Rozpoczęcie analizy opisowej...")

    df_known = df[df['Attrition'].notna()].copy()
    print(f"[stage2] Wiersze z wypełnionym Attrition: {len(df_known)} / {len(df)}")

    CHARTS_STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    DATA_STAGE2_DIR.mkdir(parents=True, exist_ok=True)

    plot_attrition_distribution(df_known)
    compute_descriptive_stats(df_known)
    top_cols = _get_top_numeric_cols(df_known)
    plot_histograms_by_attrition(df_known, top_cols)
    plot_boxplots_by_attrition(df_known, top_cols)
    run_statistical_tests(df_known)

    print("[stage2] Analiza opisowa zakończona.")


# ---------------------------------------------------------------------------
# 1. Rozkład Attrition
# ---------------------------------------------------------------------------

def plot_attrition_distribution(df: pd.DataFrame) -> None:
    """Wykres słupkowy i kołowy rozkładu zmiennej Attrition."""
    counts = df['Attrition'].value_counts()
    pcts = df['Attrition'].value_counts(normalize=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = ['#4c8cbf', '#e05c5c']
    axes[0].bar(counts.index, counts.values, color=colors)
    for i, (val, cnt) in enumerate(counts.items()):
        axes[0].text(i, cnt / 2, f'{cnt}\n({pcts[val]:.1f}%)',
                     ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    axes[0].set_title('Rozkład Attrition — liczebności')
    axes[0].set_ylabel('Liczba pracowników')
    axes[0].set_xlabel('Attrition')

    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[1].set_title('Rozkład Attrition — udziały')

    fig.suptitle('Analiza zmiennej docelowej: Attrition', fontsize=13)
    plt.tight_layout()
    plt.savefig(CHARTS_STAGE2_DIR / 'attrition_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[stage2] Zapisano: attrition_distribution.png")
    print(f"[stage2] Attrition: Yes={counts.get('Yes', 0)} ({pcts.get('Yes', 0):.1f}%), "
          f"No={counts.get('No', 0)} ({pcts.get('No', 0):.1f}%)")


# ---------------------------------------------------------------------------
# 2. Statystyki opisowe
# ---------------------------------------------------------------------------

def compute_descriptive_stats(df: pd.DataFrame) -> None:
    """Oblicza i zapisuje statystyki opisowe dla zmiennych numerycznych.

    Kolumny: mean, median, std, min, max, skewness, kurtosis, CV (std/mean).
    Zapisuje do data/stage2_descriptive/descriptive_stats.csv.
    """
    numeric_cols = df.select_dtypes(include='number').columns
    rows = []

    for col in numeric_cols:
        s = df[col].dropna()
        mean = s.mean()
        rows.append({
            'kolumna': col,
            'mean': round(mean, 2),
            'median': round(s.median(), 2),
            'std': round(s.std(), 2),
            'min': round(s.min(), 2),
            'max': round(s.max(), 2),
            'skewness': round(s.skew(), 3),
            'kurtosis': round(s.kurtosis(), 3),
            'CV': round(s.std() / mean, 3) if mean != 0 else None,
        })

    stats = pd.DataFrame(rows).sort_values('kolumna')
    out_path = DATA_STAGE2_DIR / 'descriptive_stats.csv'
    stats.to_csv(out_path, index=False)
    print(f"\n[stage2] Statystyki opisowe ({len(stats)} kolumn):")
    print(stats.to_string(index=False))
    print(f"[stage2] Zapisano: {out_path}")


# ---------------------------------------------------------------------------
# 3. Wybór top N kolumn numerycznych wg Mann-Whitney U (p-value vs Attrition)
# ---------------------------------------------------------------------------

def _get_top_numeric_cols(df: pd.DataFrame) -> list[str]:
    """Zwraca kolumny numeryczne istotnie związane z Attrition (p-value < 0.05).

    Kryterium: test Manna-Whitneya — kolumny posortowane wg p-value rosnąco.
    Wybór oparty na danych — bez hardcodu.
    """
    yes = df[df['Attrition'] == 'Yes']
    no = df[df['Attrition'] == 'No']
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    p_values = {}
    for col in numeric_cols:
        a, b = yes[col].dropna(), no[col].dropna()
        if len(a) > 0 and len(b) > 0:
            _, p = mannwhitneyu(a, b, alternative='two-sided')
            p_values[col] = p

    significant_cols = sorted(
        (col for col, p in p_values.items() if p < 0.05),
        key=p_values.get
    )
    print(f"[stage2] Kolumny numeryczne z p<0.05 ({len(significant_cols)}): {significant_cols}")
    return significant_cols


# ---------------------------------------------------------------------------
# 4. Histogramy wg Attrition
# ---------------------------------------------------------------------------

def plot_histograms_by_attrition(df: pd.DataFrame, cols: list[str]) -> None:
    """Histogramy top N kolumn numerycznych (wg Mann-Whitney) z podziałem na Attrition."""

    cols_per_row = 5
    n_rows = 3
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(25, 5 * n_rows))
    axes = axes.flatten()

    colors = {'Yes': '#e05c5c', 'No': '#4c8cbf'}

    for i, col in enumerate(cols):
        for attrition_val, color in colors.items():
            subset = df[df['Attrition'] == 'Yes'][col].dropna()
            axes[i].hist(subset, bins=25, alpha=0.6, color=color,
                         label=f'Attrition={attrition_val}', density=True)
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Gęstość')
        axes[i].legend(fontsize=8)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Rozkłady zmiennych numerycznych wg Attrition', fontsize=13)
    plt.tight_layout()
    plt.savefig(CHARTS_STAGE2_DIR / 'histograms_by_attrition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[stage2] Zapisano: histograms_by_attrition.png")


# ---------------------------------------------------------------------------
# 5. Violin ploty wg Attrition
# ---------------------------------------------------------------------------

def plot_boxplots_by_attrition(df: pd.DataFrame, cols: list[str]) -> None:
    """Violin ploty kolumn numerycznych (wg Mann-Whitney) z podziałem na Attrition.

    Violin plot pokazuje pełny kształt rozkładu (gęstość) — lepszy niż boxplot
    dla zmiennych dyskretnych gdzie mediana zlewa się z krawędzią pudełka.
    inner='box' rysuje wewnątrz miniaturowy boxplot z medianą.
    """
    cols_per_row = 5
    n_rows = 3
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(25, 5 * n_rows))
    axes = axes.flatten()

    palette = {'Yes': '#e05c5c', 'No': '#4c8cbf'}

    for i, col in enumerate(cols):
        sns.violinplot(data=df, x='Attrition', y=col, hue='Attrition',
                       palette=palette, legend=False, inner='box', ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel('Attrition')
        axes[i].set_ylabel(col)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Violin ploty zmiennych numerycznych wg Attrition', fontsize=13)
    plt.tight_layout()
    plt.savefig(CHARTS_STAGE2_DIR / 'violinplots_by_attrition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[stage2] Zapisano: violinplots_by_attrition.png")


# ---------------------------------------------------------------------------
# 6. Testy statystyczne + ranking
# ---------------------------------------------------------------------------

def run_statistical_tests(df: pd.DataFrame) -> None:
    """Testy Mann-Whitney U (numeryczne) i Chi-kwadrat (kategoryczne) vs Attrition.

    Wyniki: tabela posortowana wg p-value + wykres rankingu.
    Zapisuje data/stage2_descriptive/statistical_tests.csv.
    """
    yes = df[df['Attrition'] == 'Yes']
    no = df[df['Attrition'] == 'No']

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = [c for c in df.select_dtypes(include='object').columns
                        if c != 'Attrition']

    results = []

    for col in numeric_cols:
        a = yes[col].dropna()
        b = no[col].dropna()
        if len(a) > 0 and len(b) > 0:
            _, p = mannwhitneyu(a, b, alternative='two-sided')
            results.append({'kolumna': col, 'test': 'Mann-Whitney U', 'p_value': round(p, 6)})

    for col in categorical_cols:
        ct = pd.crosstab(df[col], df['Attrition'])
        if ct.shape[1] == 2:
            _, p, _, _ = chi2_contingency(ct)
            results.append({'kolumna': col, 'test': 'Chi-kwadrat', 'p_value': round(p, 6)})

    results_df = pd.DataFrame(results).sort_values('p_value')
    results_df['istotne'] = results_df['p_value'] < 0.05

    out_path = DATA_STAGE2_DIR / 'statistical_tests.csv'
    results_df.to_csv(out_path, index=False)

    print(f"\n[stage2] Wyniki testów statystycznych (posortowane wg p-value):")
    print(results_df.to_string(index=False))
    print(f"[stage2] Zapisano: {out_path}")
    print(f"[stage2] Istotnych zmiennych (p<0.05): {results_df['istotne'].sum()} / {len(results_df)}")

    _plot_tests_ranking(results_df)


def _plot_tests_ranking(results_df: pd.DataFrame) -> None:
    """Wykres słupkowy rankingu zmiennych wg -log10(p-value).

    Wartości -log10(p) są ograniczone do max 15 żeby skala była czytelna
    i linia p=0.05 (≈1.3) była wyraźnie widoczna. Słupki przekraczające
    próg oznaczone są gwiazdką w etykiecie.
    """
    MAX_LOG = 15
    threshold = -np.log10(0.05)

    top = results_df.head(22).copy()
    top['-log10(p)'] = (-np.log10(top['p_value'].clip(lower=1e-300))).clip(upper=MAX_LOG)
    top['label'] = top['kolumna'] + top['p_value'].apply(
        lambda p: ' ★' if p < 10**(-MAX_LOG) else ''
    )
    top = top.sort_values('-log10(p)')

    colors = ['#e05c5c' if sig else '#aaaaaa' for sig in top['istotne']]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top['label'], top['-log10(p)'], color=colors)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
               label=f'p = 0.05 (próg istotności)')
    ax.set_xlabel('-log₁₀(p-value)')
    ax.set_xlim(0, MAX_LOG + 0.5)
    ax.set_title('Ranking istotności zmiennych względem Attrition\n'
                 '(czerwony = p < 0.05, ★ = p < 1e-15)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_STAGE2_DIR / 'statistical_tests_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[stage2] Zapisano: statistical_tests_ranking.png")
