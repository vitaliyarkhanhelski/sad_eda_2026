"""
Czyszczenie danych — Osoba 1

Zakres:
- Usunięcie kolumn zbędnych:
    * Stałe (jedna wartość): EmployeeCount, Over18, StandardHours
    * Identyfikator (unikalne ID): EmployeeNumber
- Analiza i wizualizacja wzorców braków danych (MCAR / MAR / MNAR)
- Imputacja wartości brakujących:
    * Age — imputacja grupowa medianą wg TotalWorkingYears + JobLevel + JobRole
    * MonthlyIncome — imputacja grupową medianą wg JobLevel + JobRole + TotalWorkingYears
    * Attrition — NIE imputować (zmienna docelowa)
- Usuwanie duplikatów
- Wykrycie i wizualizacja wartości odstających (IQR + boxploty)
- Zapis oczyszczonego pliku HR_clean.csv

Wejście: surowy df z data_loader.load_data()
Wyjście: oczyszczony DataFrame przekazywany dalej do pipeline'u
"""

import pandas as pd
from src.settings import CONSTANT_COLS, COLS_TO_DROP, DATA_PREPROCESSING_DIR, DATA_DIR
from src.stage1_preprocessing.preprocessing_plots import (
    plot_age_mutual_information,
    plot_age_correlations,
    plot_monthly_income_mutual_information,
    plot_monthly_income_correlations,
    plot_outliers_boxplots,
    plot_missing_heatmap
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Czyści dataset HR i zwraca oczyszczony DataFrame."""
    dataset_overview(df)

    # Krok 1: Usunięcie kolumn zbędnych (stałe + ID)
    df = drop_constant_cols(df)

    # Krok 2: Imputacja Age medianą wg TotalWorkingYears + JobLevel + JobRole
    df = impute_age(df)

    # Krok 3: Imputacja MonthlyIncome medianą wg JobLevel + JobRole + TotalWorkingYears
    df = impute_monthly_income(df)

    # Krok 4: Usuwanie duplikatów
    df = drop_duplicates(df)

    # Krok 5: Analiza i wizualizacja wartości odstających (IQR)
    check_outliers(df)

    # Krok 6: Zapis oczyszczonego datasetu
    df = save_clean_data(df)

    return df

def dataset_overview(df: pd.DataFrame) -> None:
    """Print initial dataset exploration: describe, head, missing values, dtypes, and unique values for object columns."""
    print(df.shape)
    print(df.describe())
    print(df.describe(include="object"))

    print("\nMissing values per column:")
    print(df.isna().sum())
    print('')

    print(df.info())
    print('')

    print("\nAttrition value counts:")
    print(df.Attrition.value_counts(dropna=False))
    print("\nAttrition (%):")
    print((df.Attrition.value_counts(normalize=True) * 100).round(1).astype(str) + "%")
    print('')

    display_unique_values_for_object_columns(df)
    print('')
    print_constant_cols_unique(df)
    analyze_missing_patterns(df)


def analyze_missing_patterns(df: pd.DataFrame) -> None:
    """Analiza wzorców braków danych (MCAR / MAR / MNAR).

    Sprawdza:
    1. Heatmapa braków — które wiersze mają NA w których kolumnach
    2. Czy wiersze z brakującym Age różnią się od reszty wg JobLevel
       (MAR: braki zależą od JobLevel → uzasadnia imputację grupową)
    3. Czy wiersze z brakującym MonthlyIncome różnią się wg JobLevel
    """
    plot_missing_heatmap(df)

    print("\n[analyze_missing_patterns] Rozkład JobLevel — Age missing vs nie-missing:")
    age_missing = df[df['Age'].isna()]['JobLevel'].value_counts(normalize=True).round(3)
    age_present = df[df['Age'].notna()]['JobLevel'].value_counts(normalize=True).round(3)
    comparison = pd.DataFrame({'Age=NaN (%)': age_missing * 100, 'Age=OK (%)': age_present * 100}).fillna(0)
    print(comparison.to_string())
    print("→ Wniosek: rozkłady różnią się (szczególnie JobLevel 3 i 5) → MAR "
          "→ imputacja grupowa medianą wg JobLevel jest uzasadniona.")

    print("\n[analyze_missing_patterns] Rozkład JobLevel — MonthlyIncome missing vs nie-missing:")
    mi_missing = df[df['MonthlyIncome'].isna()]['JobLevel'].value_counts(normalize=True).round(3)
    mi_present = df[df['MonthlyIncome'].notna()]['JobLevel'].value_counts(normalize=True).round(3)
    comparison2 = pd.DataFrame({'MonthlyIncome=NaN (%)': mi_missing * 100, 'MonthlyIncome=OK (%)': mi_present * 100}).fillna(0)
    print(comparison2.to_string())
    print("→ Wniosek: rozkłady różnią się (szczególnie JobLevel 2 i 3) → MAR "
          "→ imputacja grupowa medianą wg JobLevel + JobRole jest uzasadniona.")


def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Imputuje brakujące wartości Age medianą grupową wg TotalWorkingYears + JobLevel + JobRole.

    Uzasadnienie (patrz: plot_age_correlations()):
    - korelacja Age ~ TotalWorkingYears = 0.684 (silna, dodatnia)
    - korelacja Age ~ JobLevel          = 0.509 (umiarkowana, dodatnia)
    - mediany Age wg JobRole różnią się od 30 (Sales Representative) do 46.5 (Manager)
    - kombinacja trzech zmiennych daje najbardziej precyzyjną imputację grupową

    Mediana zamiast średniej — odporna na wartości odstające w grupie.
    """
    plot_age_mutual_information(df)
    plot_age_correlations(df)

    print_age_missing_with_medians(df)
    df = fill_age_with_median(df)

    return df


def impute_monthly_income(df: pd.DataFrame) -> pd.DataFrame:
    """Imputuje brakujące wartości MonthlyIncome medianą grupową wg JobLevel + JobRole + TotalWorkingYears.

    Uzasadnienie (wg Mutual Information):
    - JobLevel           — MI najwyższe (>1.0), Pearson = 0.951
    - JobRole            — MI drugie miejsce (rola determinuje zakres pensji)
    - TotalWorkingYears  — MI trzecie miejsce, Pearson = 0.771
    - std wewnątrz grupy JobLevel+JobRole = 995 USD → TotalWorkingYears redukuje rozproszenie
    - 26 grup (JobLevel+JobRole) vs 309 grup (+TotalWorkingYears) — znacznie precyzyjniejsza imputacja

    Strategia: JobLevel + JobRole + TotalWorkingYears (primary), fallback: JobLevel + JobRole.
    Mediana zamiast średniej — odporna na outliery wynagrodzeń w grupie.
    """
    plot_monthly_income_mutual_information(df)
    plot_monthly_income_correlations(df)
    print_monthly_income_missing_with_medians(df)
    df = fill_monthly_income_with_median(df)
    return df


def fill_monthly_income_with_median(df: pd.DataFrame) -> pd.DataFrame:
    """Uzupełnia braki MonthlyIncome medianą grupową wg JobLevel + JobRole + TotalWorkingYears.

    Fallback: jeśli grupa ma tylko 1 wiersz z NaN, używa mediany wg JobLevel + JobRole.
    """
    missing_before = df['MonthlyIncome'].isna().sum()
    df['MonthlyIncome'] = df.groupby(['JobLevel', 'JobRole', 'TotalWorkingYears'])['MonthlyIncome'].transform(
        lambda x: x.fillna(x.median())
    )
    # fallback: jeśli grupa miała tylko 1 wiersz (sam NaN), użyj mediany wg JobLevel + JobRole
    df['MonthlyIncome'] = df.groupby(['JobLevel', 'JobRole'])['MonthlyIncome'].transform(
        lambda x: x.fillna(x.median())
    )
    missing_after = df['MonthlyIncome'].isna().sum()
    print(f"[impute_monthly_income] Uzupełniono {missing_before - missing_after} braków w MonthlyIncome "
          f"(pozostało: {missing_after})")
    return df


def print_monthly_income_missing_with_medians(df: pd.DataFrame) -> None:
    """Wyświetla wiersze z brakującym MonthlyIncome wraz z medianą do imputacji.

    Strategia: JobLevel + JobRole + TotalWorkingYears (primary), fallback: JobLevel + JobRole.
    """
    mi_medians = df.groupby(['JobLevel', 'JobRole', 'TotalWorkingYears'])['MonthlyIncome'].median()
    mi_medians_fallback = df.groupby(['JobLevel', 'JobRole'])['MonthlyIncome'].median()

    def get_monthly_income_median(row):
        v = mi_medians.get((row['JobLevel'], row['JobRole'], row['TotalWorkingYears']))
        if pd.isna(v) or v is None:
            v = mi_medians_fallback.get((row['JobLevel'], row['JobRole']))
        return v

    df_missing = df[df['MonthlyIncome'].isna()][['MonthlyIncome', 'JobLevel', 'JobRole', 'TotalWorkingYears']].copy()
    df_missing['MonthlyIncome_median_imputed'] = df_missing.apply(get_monthly_income_median, axis=1)
    print("\n[impute_monthly_income] Wiersze z brakującym MonthlyIncome:")
    print(df_missing.to_string())
    DATA_PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    df_missing.to_csv(DATA_PREPROCESSING_DIR / "monthly_income_missing.csv", index=True)


def fill_age_with_median(df: pd.DataFrame) -> pd.DataFrame:
    """Uzupełnia braki Age medianą grupową wg TotalWorkingYears + JobLevel + JobRole.

    Fallback: jeśli grupa ma tylko 1 wiersz z NaN, używa mediany wg JobLevel + JobRole.
    """
    missing_before = df['Age'].isna().sum()
    df['Age'] = df.groupby(['TotalWorkingYears', 'JobLevel', 'JobRole'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    # fallback: jeśli grupa miała tylko 1 wiersz (sam NaN), użyj mediany wg JobLevel + JobRole
    df['Age'] = df.groupby(['JobLevel', 'JobRole'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    missing_after = df['Age'].isna().sum()
    print(f"[impute_age] Uzupełniono {missing_before - missing_after} braków w Age "
          f"(pozostało: {missing_after})")
    return df


def print_age_missing_with_medians(df: pd.DataFrame) -> None:
    """Wyświetla wiersze z brakującym Age wraz z medianą która zostanie użyta do imputacji."""
    age_medians = df.groupby(['TotalWorkingYears', 'JobLevel', 'JobRole'])['Age'].median()
    age_medians_fallback = df.groupby(['JobLevel', 'JobRole'])['Age'].median()

    def get_age_median(row):
        v = age_medians.get((row['TotalWorkingYears'], row['JobLevel'], row['JobRole']))
        if pd.isna(v) or v is None:
            v = age_medians_fallback.get((row['JobLevel'], row['JobRole']))
        return v

    df_missing = df[df['Age'].isna()][['Age', 'TotalWorkingYears', 'JobLevel', 'JobRole']].copy()
    df_missing['Age_median_imputed'] = df_missing.apply(get_age_median, axis=1)
    print("\n[impute_age] Wiersze z brakującym Age:")
    print(df_missing.to_string())
    DATA_PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    df_missing.to_csv(DATA_PREPROCESSING_DIR / "age_missing.csv", index=True)



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
    - compute_outliers_summary() — tabela IQR + zapis CSV
    - plot_outliers_boxplots()   — boxploty dla zmiennych ciągłych
    """
    compute_outliers_summary(df)
    plot_outliers_boxplots(df)


def compute_outliers_summary(df: pd.DataFrame) -> None:
    """Oblicza tabelę outlierów metodą IQR dla wszystkich zmiennych numerycznych.

    Generuje tabelę: kolumna | Q1 | Q3 | IQR | dolna granica | górna granica | liczba outlierów
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

    DATA_PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PREPROCESSING_DIR / "outliers_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"[check_outliers] Zapisano raport do: {output_path}")

    cols_with_outliers = summary[summary['outlierów'] > 0]['kolumna'].tolist()
    print(f"\n[check_outliers] Kolumny z outlierami ({len(cols_with_outliers)}): {cols_with_outliers}")


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

