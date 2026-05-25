"""
Czyszczenie danych — Osoba 1

Zakres:
- Usunięcie kolumn zbędnych:
    * Stałe (jedna wartość): EmployeeCount, Over18, StandardHours
    * Identyfikator (unikalne ID): EmployeeNumber
- Analiza i wizualizacja wzorców braków danych (MCAR / MAR / MNAR)
- Imputacja wartości brakujących:
    * Age — KNN Imputer (One-Hot + RobustScaler, K wybrany metodą łokcia)
    * MonthlyIncome — KNN Imputer (One-Hot + RobustScaler, K wybrany metodą łokcia)
    * Attrition — NIE imputować (zmienna docelowa)
- Usuwanie duplikatów
- Wykrycie i wizualizacja wartości odstających (IQR + boxploty)
- Zapis oczyszczonego pliku HR_clean.csv

Wejście: surowy df z data_loader.load_data()
Wyjście: oczyszczony DataFrame przekazywany dalej do pipeline'u
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, shapiro
from skimpy import skim
import os
os.environ["COLUMNS"] = "200"
from src.settings import CONSTANT_COLS, COLS_TO_DROP, DATA_DIR, DATA_STAGE1_DIR
from src.plots import (
    msno_plots,
    plot_age_imputation_analysis,
    plot_monthly_income_imputation_analysis,
    plot_outliers_boxplots
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Czyści dataset HR i zwraca oczyszczony DataFrame."""

    # Krok 1: Usuwanie duplikatów (przed imputacją)
    df = drop_duplicates(df)

    # Krok 2: Usunięcie kolumn zbędnych (stałe + ID)
    df = drop_constant_cols(df)

    # Krok 3: Imputacja Age
    df = impute_age(df)

    # Krok 4: Imputacja MonthlyIncome
    df = impute_monthly_income(df)

    # Krok 5: Analiza i wizualizacja wartości odstających (IQR)
    check_outliers(df)

    # Krok 6: Zapis oczyszczonego datasetu
    df = save_clean_data(df)

    return df


def dataset_analysis(df: pd.DataFrame) -> None:
    """Pełna eksploracja datasetu: wizualizacje braków + overview statystyczny."""
    msno_plots(df)
    dataset_overview(df)


def dataset_overview(df: pd.DataFrame) -> None:
    """Print initial dataset exploration: describe, head, missing values, dtypes, and unique values for object columns."""
    skim(df)

    print("\nAttrition value counts:")
    print(df.Attrition.value_counts(dropna=False))
    print("\nAttrition (%):")
    print((df.Attrition.value_counts(normalize=True, dropna=False) * 100).round(1).astype(str) + "%")
    print('')

    display_unique_values_for_object_columns(df)
    print('')
    print_constant_cols_unique(df)
    check_duplicates_overview(df)

    # Krok 1: MI — które zmienne są powiązane z Age i MonthlyIncome (uzasadnienie wyboru features do imputacji)
    plot_age_imputation_analysis(df)
    plot_monthly_income_imputation_analysis(df)

    # Krok 2: Analiza wzorców braków — czy braki zależą od odkrytych zmiennych (MAR/MCAR)
    analyze_missing_patterns(df)


def check_duplicates_overview(df: pd.DataFrame) -> None:
    """Sprawdza duplikaty w datasecie i wyświetla wyniki."""
    n_duplicates = df.duplicated().sum()
    print(f"\nDuplikaty (wszystkie kolumny): {n_duplicates}")
    if n_duplicates > 0:
        print(df[df.duplicated(keep=False)].sort_values(list(df.columns)).to_string())
    else:
        print("Brak duplikatów.")



def analyze_missing_patterns(df: pd.DataFrame) -> None:
    """Analiza wzorców braków danych (MCAR / MAR / MNAR).

    Sprawdza:
    1. Age ~ TotalWorkingYears  — Mann-Whitney U (zmienna ciągła, top MI dla Age)
    2. Age ~ JobLevel           — Chi-square (zmienna kategoryczna)
    3. MonthlyIncome ~ JobLevel — Chi-square (top MI dla MonthlyIncome)
    """
    _test_mar_mannwhitney(df, missing_col='Age', feature_col='TotalWorkingYears')
    _test_mar_chi2(df, missing_col='Age', feature_col='JobLevel')
    _test_mar_chi2(df, missing_col='MonthlyIncome', feature_col='JobLevel')


def _test_mar_mannwhitney(df: pd.DataFrame, missing_col: str, feature_col: str) -> None:
    """Shapiro-Wilk → Mann-Whitney U: czy rozkład feature_col różni się między grupą z brakiem a bez.

    Shapiro-Wilk uzasadnia wybór testu nieparametrycznego (Mann-Whitney zamiast t-test).
    Uwaga: przy n > 5000 Shapiro-Wilk traci moc — używamy próbki 5000.
    """
    _test_normality(df, feature_col)
    _run_mannwhitney(df, missing_col, feature_col)


def _run_mannwhitney(df: pd.DataFrame, missing_col: str, feature_col: str) -> None:
    """Mann-Whitney U: porównuje rozkład feature_col między grupą z brakiem a bez."""
    group_missing = df[df[missing_col].isna()][feature_col].dropna()
    group_present = df[df[missing_col].notna()][feature_col].dropna()
    stat, p = mannwhitneyu(group_missing, group_present, alternative='two-sided')
    conclusion = "MAR (rozkłady różnią się)" if p < 0.05 else "brak dowodów na MAR (MCAR możliwe)"
    print(f"\n[analyze_missing_patterns] Mann-Whitney U: {missing_col} ~ {feature_col}")
    print(f"  n(missing)={len(group_missing)}, n(present)={len(group_present)}")
    print(f"  U={stat:.1f}, p={p:.4f} → {conclusion}")


def _test_normality(df: pd.DataFrame, feature_col: str) -> None:
    """Shapiro-Wilk: sprawdza normalność rozkładu feature_col.

    Uzasadnia wybór testu parametrycznego (t-test) lub nieparametrycznego (Mann-Whitney).
    Uwaga: przy n > 5000 Shapiro-Wilk traci moc — używamy próbki 5000.
    """
    data = df[feature_col].dropna()
    sample = data.sample(min(len(data), 5000), random_state=42)
    sw_stat, sw_p = shapiro(sample)
    normal = sw_p >= 0.05
    print(f"\n[analyze_missing_patterns] Shapiro-Wilk: {feature_col}")
    print(f"  W={sw_stat:.4f}, p={sw_p:.4f} → {'normalny' if normal else 'NIE-normalny'} rozkład"
          f" → {'t-test' if normal else 'Mann-Whitney U (nieparametryczny)'}")


def _test_mar_chi2(df: pd.DataFrame, missing_col: str, feature_col: str) -> None:
    """Chi-square: czy rozkład feature_col jest niezależny od missingness w missing_col."""
    missing_flag = df[missing_col].isna().astype(int)
    contingency = pd.crosstab(missing_flag, df[feature_col])
    chi2, p, dof, _ = chi2_contingency(contingency)
    conclusion = "MAR (zależność istotna)" if p < 0.05 else "brak dowodów na MAR (MCAR możliwe)"
    print(f"\n[analyze_missing_patterns] Chi-square: {missing_col} ~ {feature_col}")
    print(f"  chi2={chi2:.3f}, dof={dof}, p={p:.4f} → {conclusion}")
    print(contingency.to_string())


def _prepare_knn_features(df: pd.DataFrame, target: str) -> tuple:
    """Przygotowuje macierz cech do KNN Imputer: One-Hot + RobustScaler.

    Kolumny kategoryczne (object) → pd.get_dummies (One-Hot).
    Kolumna 'Attrition' jest pomijana (zmienna docelowa Y, nie feature).
    Wszystkie wartości numeryczne są skalowane RobustScaler (odpornym na outliery).
    KNNImputer poradzi sobie z NaN w kolumnach cech (np. MonthlyIncome podczas imputacji Age).

    Parameters
    ----------
    df : pd.DataFrame
        Surowy DataFrame (może zawierać NaN w kolumnach cech).
    target : str
        Kolumna imputowana — zostaje w macierzy (KNNImputer widzi NaN i uzupełnia).

    Returns
    -------
    X_scaled : np.ndarray
        Przeskalowana macierz cech gotowa do KNNImputer.
    scaler : RobustScaler
        Dopasowany scaler (potrzebny do odwrócenia skalowania po imputacji).
    target_idx : int
        Indeks kolumny `target` w X_scaled.
    """
    from sklearn.preprocessing import RobustScaler

    df_work = df.drop(columns=['Attrition'], errors='ignore')

    cat_cols = df_work.select_dtypes(include='object').columns.tolist()
    df_enc = pd.get_dummies(df_work, columns=cat_cols, drop_first=False)

    target_idx = df_enc.columns.tolist().index(target)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_enc.values.astype(float))

    return X_scaled, scaler, target_idx


def plot_knn_elbow_chart(
    X_scaled: np.ndarray,
    target_idx: int,
    target: str,
    k_range: range = range(1, 21),
) -> None:
    """Generuje wykres łokcia MSE vs K dla KNN imputation.

    Bierze 20% znanych wierszy, sztucznie usuwa target, uruchamia KNNImputer
    dla każdego K i liczy MSE. Wykres zapisywany do charts/knn/.
    K wybieramy ręcznie na oko z wykresu i hardcodujemy w impute_age / impute_monthly_income.

    Parameters
    ----------
    X_scaled : np.ndarray
        Przeskalowana macierz cech (z NaN w kolumnie target dla prawdziwych braków).
    target_idx : int
        Indeks kolumny target w X_scaled.
    target : str
        Nazwa kolumny (do wykresu).
    k_range : range
        Zakres K do przetestowania.
    """
    from sklearn.impute import KNNImputer
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from src.plots import plot_knn_elbow

    target_values = X_scaled[:, target_idx]
    known_idx = np.where(~np.isnan(target_values))[0]
    _, test_idx = train_test_split(known_idx, test_size=0.2, random_state=42)

    true_values = X_scaled[test_idx, target_idx].copy()

    mse_scores = []
    for k in k_range:
        X_sim = X_scaled.copy()
        X_sim[test_idx, target_idx] = np.nan
        X_imp = KNNImputer(n_neighbors=k).fit_transform(X_sim)
        mse_scores.append(mean_squared_error(true_values, X_imp[test_idx, target_idx]))

    plot_knn_elbow(mse_scores, k_range, target)


def _impute_with_knn(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    scaler,
    target_idx: int,
    target: str,
    k_opt: int,
) -> pd.Series:
    """Wykonuje finalną imputację KNN używając już przygotowanej macierzy cech.

    X_scaled i scaler są przekazywane z zewnątrz — obliczone raz w impute_age/impute_monthly_income.

    Returns
    -------
    pd.Series z uzupełnionymi wartościami (tylko braki zastąpione, reszta niezmieniona).
    """
    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=k_opt)
    X_imputed = imputer.fit_transform(X_scaled)

    X_inv = scaler.inverse_transform(X_imputed)
    imputed_col = pd.Series(X_inv[:, target_idx], index=df.index)

    result = df[target].copy()
    missing_mask = result.isna()
    result[missing_mask] = imputed_col[missing_mask]
    return result



def _save_imputed_rows(df_before: pd.DataFrame, df_after: pd.DataFrame, target: str, context_cols: list[str]) -> None:
    """Zapisuje wiersze, które miały NaN w `target`, z wartością wstawioną przez KNN.

    Kolumny w pliku: context_cols + Age_knn_imputed (lub MonthlyIncome_knn_imputed).
    Plik: data/stage1_preprocessing/{target_lower}_imputed_rows.csv
    """
    missing_mask = df_before[target].isna()
    result = df_before.loc[missing_mask, context_cols].copy()
    result[f'{target}_knn_imputed'] = df_after.loc[missing_mask, target]

    DATA_STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_STAGE1_DIR / f"{target.lower()}_imputed_rows.csv"
    result.to_csv(out_path)
    print(f"[impute_{target.lower()}] Zapisano wiersze po imputacji ({missing_mask.sum()} wierszy): {out_path}")


def _print_imputation_stats(before: pd.Series, after: pd.Series, target: str) -> None:
    """Drukuje i zapisuje tabelę statystyk przed i po imputacji.

    Porównuje mean, std, median — wartości powinny pozostać zbliżone.
    Zapisuje CSV do data/stage1_preprocessing/{target_lower}_imputation_stats.csv.
    """
    stats = pd.DataFrame({
        'Przed (znane)': before.dropna().describe()[['mean', 'std', '50%']],
        'Po KNN': after.describe()[['mean', 'std', '50%']],
    }).rename(index={'mean': 'Mean (średnia)', 'std': 'Std (odchylenie std)', '50%': 'Median (mediana)'})

    print(f"\n[impute_{target.lower()}] Statystyki przed/po imputacji:")
    print(stats.round(2).to_string())

    DATA_STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_STAGE1_DIR / f"{target.lower()}_imputation_stats.csv"
    stats.to_csv(out_path)
    print(f"[impute_{target.lower()}] Zapisano: {out_path}")


def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Imputuje brakujące wartości Age metodą KNN.

    Uzasadnienie wyboru cech (patrz: plot_age_imputation_analysis()):
    - TotalWorkingYears — Mutual Information i Pearson=0.684 (najsilniejsza korelacja)
    - JobLevel          — Pearson=0.509 (umiarkowana, dodatnia)
    - YearsAtCompany    — wysoka MI i korelacja
    - pozostałe cechy numeryczne + One-Hot kategorycznych

    Pipeline:
    1. Przygotowanie cech: One-Hot (kategoryczne) + RobustScaler (numeryczne)
    2. Wybór K metodą łokcia (MSE na sztucznie usuniętych wartościach) → wykres age_knn_elbow.png
    3. Finalna imputacja KNNImputer z optymalnym K
    """
    k = 6  # wybrane na podstawie wykresu łokcia (charts/knn/age_knn_elbow.png)
    print(f"\n[impute_age] Braków Age przed imputacją: {df['Age'].isna().sum()}, K={k}")

    X_scaled, scaler, target_idx = _prepare_knn_features(df, 'Age')
    plot_knn_elbow_chart(X_scaled, target_idx, 'Age')
    df_before = df.copy()
    age_before = df['Age'].copy()
    df['Age'] = _impute_with_knn(df, X_scaled, scaler, target_idx, 'Age', k)
    df['Age'] = df['Age'].round().astype(int)
    _save_imputed_rows(df_before, df, 'Age', ['TotalWorkingYears', 'JobLevel', 'YearsAtCompany', 'JobRole'])
    _print_imputation_stats(age_before, df['Age'], 'Age')

    print(f"[impute_age] Braków Age po imputacji: {df['Age'].isna().sum()}")
    return df


def impute_monthly_income(df: pd.DataFrame) -> pd.DataFrame:
    """Imputuje brakujące wartości MonthlyIncome metodą KNN.

    Uzasadnienie wyboru cech (patrz: plot_monthly_income_imputation_analysis()):
    - JobLevel          — MI najwyższe (>1.0), Pearson=0.951 (dominująca zależność)
    - JobRole           — MI drugie miejsce (rola determinuje zakres pensji)
    - TotalWorkingYears — MI trzecie miejsce, Pearson=0.771

    Pipeline:
    1. Przygotowanie cech: One-Hot (kategoryczne) + RobustScaler (numeryczne)
    2. Wybór K metodą łokcia (MSE na sztucznie usuniętych wartościach) → wykres monthly_income_knn_elbow.png
    3. Finalna imputacja KNNImputer z optymalnym K
    """
    k = 4  # wybrane na podstawie wykresu łokcia (charts/knn/monthly_income_knn_elbow.png)
    print(f"\n[impute_monthly_income] Braków MonthlyIncome przed imputacją: {df['MonthlyIncome'].isna().sum()}, K={k}")

    X_scaled, scaler, target_idx = _prepare_knn_features(df, 'MonthlyIncome')
    plot_knn_elbow_chart(X_scaled, target_idx, 'MonthlyIncome')
    df_before = df.copy()
    income_before = df['MonthlyIncome'].copy()
    df['MonthlyIncome'] = _impute_with_knn(df, X_scaled, scaler, target_idx, 'MonthlyIncome', k)
    df['MonthlyIncome'] = df['MonthlyIncome'].round().astype(int)
    _save_imputed_rows(df_before, df, 'MonthlyIncome', ['JobLevel', 'JobRole', 'TotalWorkingYears'])
    _print_imputation_stats(income_before, df['MonthlyIncome'], 'MonthlyIncome')

    print(f"[impute_monthly_income] Braków MonthlyIncome po imputacji: {df['MonthlyIncome'].isna().sum()}")
    return df



def save_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Zapisuje oczyszczony dataset do data/HR_clean.csv."""
    output_path = DATA_DIR / "HR_clean.csv"
    df.to_csv(output_path, index=False)
    print(f"[save_clean_data] Zapisano oczyszczony dataset: {output_path} "
          f"({df.shape[0]} wierszy × {df.shape[1]} kolumn)")
    return df


def check_outliers(df: pd.DataFrame) -> None:
    """Wykrywa i wizualizuje wartości odstające metodą IQR.

    Wywołuje:
    - compute_outliers_summary() — tabela IQR + zapis CSV, zwraca posortowany DataFrame
    - plot_outliers_boxplots()   — boxploty top 6 kolumn wg liczby outlierów
    """
    summary = compute_outliers_summary(df)
    plot_outliers_boxplots(df, summary)


def compute_outliers_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Oblicza tabelę outlierów metodą IQR dla wszystkich zmiennych numerycznych.

    Generuje tabelę: kolumna | Q1 | Q3 | IQR | dolna granica | górna granica | liczba outlierów
    Zwraca DataFrame posortowany malejąco wg liczby outlierów.
    Nie usuwa outlierów — dane HR mogą zawierać skrajne ale prawidłowe wartości.
    """
    numeric_cols = df.select_dtypes(include='number').columns
    results = []

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        results.append({
            'kolumna': col,
            'Q1': round(q1, 2),
            'Q3': round(q3, 2),
            'IQR': round(iqr, 2),
            'dolna_granica': round(lower, 2),
            'górna_granica': round(upper, 2),
            'outlierów': n_outliers,
            '% outlierów': round(n_outliers / len(df) * 100, 1)
        })

    summary = pd.DataFrame(results).sort_values('outlierów', ascending=False)
    print("\n[check_outliers] Outliery wg metody IQR:")
    print(summary.to_string(index=False))

    DATA_STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_STAGE1_DIR / "outliers_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"[check_outliers] Zapisano raport do: {output_path}")
    return summary


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Sprawdza i usuwa zduplikowane wiersze.

    Sprawdza duplikaty na dwa sposoby:
    - pełne duplikaty (wszystkie kolumny)
    - duplikaty bez uwzględnienia EmployeeNumber (może już być usunięty w Kroku 1)
    """
    full_dupes = df.duplicated().sum()
    print(f"[drop_duplicates] Duplikaty (wszystkie kolumny): {full_dupes}")
    if full_dupes == 0:
        print("[drop_duplicates] Brak duplikatów — pomijam.")
        return df

    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[drop_duplicates] Usunięto {before - after} duplikatów (pozostało: {after} wierszy)")
    return df


def drop_constant_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Usuwa zbędne kolumny: stałe (jedna wartość) + EmployeeNumber (unikalny ID)."""
    df = df.drop(columns=COLS_TO_DROP)
    print(f"[preprocessing] Usunięto kolumny: {COLS_TO_DROP}")
    return df


def display_unique_values_for_object_columns(df: pd.DataFrame) -> None:
    """Print unique values for each column of object (string) type."""
    for col in df.select_dtypes(include="object").columns:
        print(f"{col}: {df[col].unique()}")


def print_constant_cols_unique(df: pd.DataFrame) -> None:
    """Print unique values for constant columns defined in settings.CONSTANT_COLS."""
    for col in CONSTANT_COLS:
        print(f"{col}: {df[col].unique()}")

