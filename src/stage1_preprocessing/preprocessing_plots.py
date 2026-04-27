"""
Wizualizacje dla etapu czyszczenia danych — Etap 1

Wykresy uzasadniające decyzje imputacji i czyszczenia.
Wyniki zapisywane do charts/1_preprocessing/.
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from src.settings import CHARTS_PREPROCESSING_DIR, CONTINUOUS_COLS_FOR_OUTLIERS

try:
    matplotlib.use('TkAgg')  # interaktywne okna w JetBrains Python Console
except Exception:
    pass  # fallback — w Jupyter i środowiskach bez GUI wykresy zapisują się tylko do pliku


def plot_missing_heatmap(df: pd.DataFrame) -> None:
    """Heatmapa braków danych — które wiersze mają NA w których kolumnach.

    Wizualizuje wzorzec braków (MCAR/MAR/MNAR).
    Pokazuje tylko kolumny które mają co najmniej jeden brak.
    """
    cols_with_missing = df.columns[df.isna().any()].tolist()
    if not cols_with_missing:
        print("[plot_missing_heatmap] Brak wartości NA — pomijam heatmapę.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[cols_with_missing].isna(), cbar=False, cmap='viridis',
                yticklabels=False, ax=ax)
    ax.set_title('Heatmapa braków danych\n(żółty = NaN, fioletowy = wartość)', fontsize=12)
    ax.set_xlabel('Kolumny')

    plt.tight_layout()
    CHARTS_PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHARTS_PREPROCESSING_DIR / "missing_heatmap.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_mutual_information(df: pd.DataFrame, target: str) -> None:
    """Generyczny wykres Mutual Information — ważność zmiennych względem dowolnego targetu.

    Wyklucza automatycznie: target oraz 'Attrition' (zmienna docelowa projektu).
    Zmienne kategoryczne kodowane jako kody liczbowe (tylko do celów MI).

    Args:
        df: DataFrame z danymi
        target: nazwa kolumny docelowej (np. 'Age', 'MonthlyIncome')
    """
    df_mi = df.dropna(subset=[target]).copy()

    for col in df_mi.select_dtypes(include='object').columns:
        df_mi[col] = df_mi[col].astype('category').cat.codes

    X = df_mi.drop(columns=[target, 'Attrition']).dropna(axis=1)
    y = df_mi[target]

    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    mi_series.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'Mutual Information — ważność zmiennych względem {target}\n(uzasadnienie wyboru zmiennych do imputacji)', fontsize=12)
    ax.set_xlabel('Mutual Information score')

    plt.tight_layout()
    CHARTS_PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHARTS_PREPROCESSING_DIR / f"{target.lower()}_mutual_information.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[plot_mutual_information] Top 5 zmiennych dla {target}:")
    print(mi_series.tail(5).sort_values(ascending=False).round(3))


def plot_outliers_boxplots(df: pd.DataFrame) -> None:
    """Boxploty dla zmiennych ciągłych z outlierami (IQR).

    Pomija zmienne dyskretne z małą wariancją (PerformanceRating, StockOptionLevel itp.)
    gdzie IQR daje fałszywe outliery.

    Kolumny zdefiniowane w settings.CONTINUOUS_COLS_FOR_OUTLIERS.
    """
    cols = CONTINUOUS_COLS_FOR_OUTLIERS

    # Oblicz liczbę outlierów dla każdej kolumny
    outlier_counts = {}
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        outlier_counts[col] = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()

    cols = sorted(cols, key=lambda c: outlier_counts[c], reverse=True)

    cols_per_row = 3
    n_rows = -(-len(cols) // cols_per_row)  # ceiling division
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor='steelblue', alpha=0.6))
        axes[i].set_title(f'{col}\n(outlierów: {outlier_counts[col]})')
        axes[i].set_ylabel(col)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Boxploty — zmienne ciągłe (analiza outlierów IQR)', fontsize=13)
    plt.tight_layout()
    CHARTS_PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHARTS_PREPROCESSING_DIR / "outliers_boxplots.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_age_mutual_information(df: pd.DataFrame) -> None:
    """Wykres Mutual Information — ważność wszystkich zmiennych względem Age."""
    plot_mutual_information(df, target='Age')


def plot_monthly_income_mutual_information(df: pd.DataFrame) -> None:
    """Wykres Mutual Information — ważność wszystkich zmiennych względem MonthlyIncome."""
    plot_mutual_information(df, target='MonthlyIncome')


def plot_imputation_correlations(df: pd.DataFrame, target: str) -> None:
    """Wizualizuje korelacje target z TotalWorkingYears, JobLevel i JobRole.

    Generyczna funkcja uzasadniająca wybór zmiennych do imputacji.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Uzasadnienie imputacji {target} — korelacje z zmiennymi grupującymi", fontsize=13)

    # 1. target vs TotalWorkingYears
    axes[0].scatter(df['TotalWorkingYears'], df[target], alpha=0.3, s=10)
    axes[0].set_xlabel('TotalWorkingYears')
    axes[0].set_ylabel(target)
    corr_twy = df[[target, 'TotalWorkingYears']].corr().iloc[0, 1].round(3)
    axes[0].set_title(f'{target} ~ TotalWorkingYears\nPearson r = {corr_twy}')

    # 2. target vs JobLevel
    axes[1].scatter(df['JobLevel'], df[target], alpha=0.3, s=10)
    axes[1].set_xlabel('JobLevel')
    axes[1].set_ylabel(target)
    corr_jl = df[[target, 'JobLevel']].corr().iloc[0, 1].round(3)
    axes[1].set_title(f'{target} ~ JobLevel\nPearson r = {corr_jl}')

    # 3. Mediana target wg JobRole (boxplot)
    order = df.groupby('JobRole')[target].median().sort_values().index
    sns.boxplot(data=df, x='JobRole', y=target, order=order, ax=axes[2])
    axes[2].tick_params(axis='x', rotation=45, labelsize=8)
    axes[2].set_title(f'{target} wg JobRole\n(mediana rośnie wraz z rolą)')

    plt.tight_layout()
    CHARTS_PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHARTS_PREPROCESSING_DIR / f"{target.lower()}_imputation_correlations.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_age_correlations(df: pd.DataFrame) -> None:
    """Wizualizuje korelacje Age z TotalWorkingYears, JobLevel i JobRole."""
    plot_imputation_correlations(df, target='Age')


def plot_monthly_income_correlations(df: pd.DataFrame) -> None:
    """Wizualizuje korelacje MonthlyIncome z TotalWorkingYears, JobLevel i JobRole."""
    plot_imputation_correlations(df, target='MonthlyIncome')


