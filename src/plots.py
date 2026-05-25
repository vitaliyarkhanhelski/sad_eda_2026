"""
Wizualizacje dla etapu czyszczenia danych — Etap 1

Wykresy uzasadniające decyzje imputacji i czyszczenia.
Wyniki zapisywane do charts/1_preprocessing/.
"""

import missingno as msno
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import numpy as np
from src.settings import CHARTS_DIR, CHARTS_MSNO_DIR, CHARTS_AGE_DIR, CHARTS_MONTHLY_INCOME_DIR, CHARTS_KNN_DIR

try:
    matplotlib.use('TkAgg')  # interaktywne okna w JetBrains Python Console
except Exception:
    pass  # fallback — w Jupyter i środowiskach bez GUI wykresy zapisują się tylko do pliku


def msno_plots(df: pd.DataFrame, sort_by: str = 'Age') -> None:
    """Generuje wszystkie wykresy braków danych (matrix, heatmapa, dendrogram, msno heatmapa).

    Wyznacza wspólną listę kolumn z brakami i przekazuje ją do wykresów.
    """
    cols_with_missing = df.columns[df.isna().any()].tolist()
    if not cols_with_missing:
        print("[msno_plots] Brak wartości NA — pomijam wykresy.")
        return
    CHARTS_MSNO_DIR.mkdir(parents=True, exist_ok=True)
    plot_msno_matrix(df, cols_with_missing, sort_by)
    plot_missing_heatmap(df, cols_with_missing)
    plot_msno_dendrogram(df)
    plot_msno_heatmap(df)
    print(
        "\n[msno_plots] Wnioski z analizy wzorców braków:"
        f"\n  • Kolumny z brakami: {cols_with_missing}"
        "\n  • msno.heatmap: korelacje nullity bliskie 0 → braki w poszczególnych kolumnach są od siebie niezależne"
        "\n  • msno.dendrogram: kolumny bez braków grupują się razem (dystans=0); kolumny z brakami łączą się osobno"
        "\n  • Mechanizm braków (MAR/MCAR) wymaga dalszej analizy statystycznej (Chi-square) — patrz analyze_missing_patterns()"
        "\n  • Attrition nie jest imputowane — zmienna docelowa (Y)"
    )


def plot_msno_matrix(df: pd.DataFrame, cols_with_missing: list, sort_by: str = 'Age') -> None:
    """Macierz braków (shadow map) posortowana wg wybranej kolumny.

    Sortowanie wg Age ujawnia strukturalny wzorzec braków — wiersze z NaN
    w Age i MonthlyIncome skupiają się razem, co sugeruje MAR (braki zależą
    od obserwowalnych zmiennych jak JobLevel), a nie MNAR.
    """
    msno.matrix(df[cols_with_missing].sort_values(by=sort_by))
    path = CHARTS_MSNO_DIR / "msno_matrix.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[plot_msno_matrix] Wykres zapisany: {path}")
    print("[plot_msno_matrix] Wzorzec braków → MAR (Missing At Random)")


def plot_missing_heatmap(df: pd.DataFrame, cols_with_missing: list) -> None:
    """Heatmapa braków danych — które wiersze mają NA w których kolumnach.

    Wizualizuje wzorzec braków (MCAR/MAR/MNAR).
    Pokazuje tylko kolumny które mają co najmniej jeden brak.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[cols_with_missing].isna(), cbar=False, cmap='viridis',
                yticklabels=False, ax=ax)
    ax.set_title('Heatmapa braków danych\n(żółty = NaN, fioletowy = wartość)', fontsize=12)
    ax.set_xlabel('Kolumny')

    plt.tight_layout()
    plt.savefig(CHARTS_MSNO_DIR / "missing_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_msno_dendrogram(df: pd.DataFrame) -> None:
    """Dendrogram braków — grupuje wszystkie kolumny wg podobieństwa wzorca braków (missingno)."""
    msno.dendrogram(df)
    path = CHARTS_MSNO_DIR / "msno_dendrogram.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[plot_msno_dendrogram] Wykres zapisany: {path}")


def plot_msno_heatmap(df: pd.DataFrame) -> None:
    """Heatmapa korelacji braków (missingno) — pokazuje czy braki w kolumnach współwystępują."""
    msno.heatmap(df)
    path = CHARTS_MSNO_DIR / "msno_heatmap.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[plot_msno_heatmap] Wykres zapisany: {path}")


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

    out_dir = CHARTS_AGE_DIR if target == 'Age' else CHARTS_MONTHLY_INCOME_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "mutual_information.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot_mutual_information] Top 5 zmiennych dla {target}:")
    print(mi_series.tail(5).sort_values(ascending=False).round(3))


def plot_outliers_boxplots(df: pd.DataFrame, summary: pd.DataFrame, top_n: int = 6, min_unique: int = 6) -> None:
    """Boxploty top N kolumn z największą liczbą outlierów.

    Kolumny i ich liczba outlierów są przekazywane z compute_outliers_summary()
    — summary jest już posortowane malejąco wg 'outlierów'.

    Pomija kolumny z małą liczbą unikalnych wartości (< min_unique), gdzie IQR
    daje fałszywe outliery (np. PerformanceRating, StockOptionLevel).
    """
    continuous_cols = summary[summary['kolumna'].map(lambda c: df[c].nunique()) >= min_unique]
    top_cols = continuous_cols.head(top_n)[['kolumna', 'outlierów']].values

    cols_per_row = 3
    n_rows = -(-top_n // cols_per_row)
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, (col, n_outliers) in enumerate(top_cols):
        axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor='steelblue', alpha=0.6))
        axes[i].set_title(f'{col}\n(outlierów: {int(n_outliers)})')
        axes[i].set_ylabel(col)

    for j in range(top_n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Boxploty — top {top_n} kolumn wg liczby outlierów (IQR)', fontsize=13)
    plt.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHARTS_DIR / "outliers_boxplots.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_age_mutual_information(df: pd.DataFrame) -> None:
    """Wykres Mutual Information — ważność wszystkich zmiennych względem Age."""
    plot_mutual_information(df, target='Age')


def plot_monthly_income_mutual_information(df: pd.DataFrame) -> None:
    """Wykres Mutual Information — ważność wszystkich zmiennych względem MonthlyIncome."""
    plot_mutual_information(df, target='MonthlyIncome')


def plot_imputation_correlations(df: pd.DataFrame, target: str) -> None:
    """Wizualizuje korelacje target z TotalWorkingYears, YearsAtCompany, JobLevel i JobRole.

    Generyczna funkcja uzasadniająca wybór zmiennych do imputacji.
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f"Uzasadnienie imputacji {target} — korelacje z zmiennymi grupującymi", fontsize=13)

    # 1. target vs TotalWorkingYears
    axes[0].scatter(df['TotalWorkingYears'], df[target], alpha=0.3, s=10)
    axes[0].set_xlabel('TotalWorkingYears')
    axes[0].set_ylabel(target)
    corr_twy = df[[target, 'TotalWorkingYears']].corr().iloc[0, 1].round(3)
    axes[0].set_title(f'{target} ~ TotalWorkingYears\nPearson r = {corr_twy}')

    # 2. target vs YearsAtCompany
    axes[1].scatter(df['YearsAtCompany'], df[target], alpha=0.3, s=10)
    axes[1].set_xlabel('YearsAtCompany')
    axes[1].set_ylabel(target)
    corr_yac = df[[target, 'YearsAtCompany']].corr().iloc[0, 1].round(3)
    axes[1].set_title(f'{target} ~ YearsAtCompany\nPearson r = {corr_yac}')

    # 3. target vs JobLevel
    axes[2].scatter(df['JobLevel'], df[target], alpha=0.3, s=10)
    axes[2].set_xlabel('JobLevel')
    axes[2].set_ylabel(target)
    corr_jl = df[[target, 'JobLevel']].corr().iloc[0, 1].round(3)
    axes[2].set_title(f'{target} ~ JobLevel\nPearson r = {corr_jl}')

    # 4. Mediana target wg JobRole (boxplot)
    order = df.groupby('JobRole')[target].median().sort_values().index
    sns.boxplot(data=df, x='JobRole', y=target, order=order, ax=axes[3])
    axes[3].tick_params(axis='x', rotation=45, labelsize=8)
    axes[3].set_title(f'{target} wg JobRole\n(mediana rośnie wraz z rolą)')

    out_dir = CHARTS_AGE_DIR if target == 'Age' else CHARTS_MONTHLY_INCOME_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "imputation_correlations.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_age_correlations(df: pd.DataFrame) -> None:
    """Wizualizuje korelacje Age z TotalWorkingYears, JobLevel i JobRole."""
    plot_imputation_correlations(df, target='Age')


def plot_monthly_income_correlations(df: pd.DataFrame) -> None:
    """Wizualizuje korelacje MonthlyIncome z TotalWorkingYears, JobLevel i JobRole."""
    plot_imputation_correlations(df, target='MonthlyIncome')


def plot_age_imputation_analysis(df: pd.DataFrame) -> None:
    """MI + korelacje dla Age — uzasadnienie wyboru zmiennych do imputacji."""
    plot_age_mutual_information(df)
    plot_age_correlations(df)


def plot_monthly_income_imputation_analysis(df: pd.DataFrame) -> None:
    """MI + korelacje dla MonthlyIncome — uzasadnienie wyboru zmiennych do imputacji."""
    plot_monthly_income_mutual_information(df)
    plot_monthly_income_correlations(df)


def plot_knn_elbow(mse_scores: list[float], k_range: range, target: str) -> None:
    """Zapisuje wykres metody łokcia (MSE vs K) dla KNN imputation.

    Parameters
    ----------
    mse_scores : list[float]
        MSE dla każdego K z k_range (liczone na sztucznie usuniętych wartościach).
    k_range : range
        Zakres testowanych wartości K (np. range(1, 21)).
    target : str
        Nazwa imputowanej kolumny ('Age' lub 'MonthlyIncome').
    """
    CHARTS_KNN_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{target.lower().replace('monthlyincome', 'monthly_income')}_knn_elbow.png"
    ks = list(k_range)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, mse_scores, marker='o', linewidth=2, color='steelblue')
    ax.set_xlabel('K (liczba sąsiadów)')
    ax.set_ylabel('MSE')
    ax.set_title(f'Metoda łokcia — KNN Imputer ({target})')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(CHARTS_KNN_DIR / filename, dpi=150)
    plt.close()
    print(f"[plot_knn_elbow] Zapisano: {CHARTS_KNN_DIR / filename}")

