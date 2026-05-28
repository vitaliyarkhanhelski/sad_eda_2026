"""
Przygotowanie danych do modelowania — Stage 5

Zakres:
- Encoding zmiennych kategorycznych:
    * Binary: Attrition (Yes=1/No=0), Gender (Male=1/Female=0), OverTime (Yes=1/No=0)
    * Ordinal: BusinessTravel (Non-Travel=0, Travel_Rarely=1, Travel_Frequently=2)
    * One-Hot: Department, EducationField, JobRole, MaritalStatus
- Skalowanie zmiennych numerycznych:
    * Standaryzacja Z-score (StandardScaler) — mean=0, std=1
    * Normalizacja Min-Max (MinMaxScaler) — zakres [0, 1]
- Wizualizacja rozkładów przed i po skalowaniu (top 4 zmienne)
- Zapis finalnych datasetów:
    * HR_model_standardized.csv
    * HR_model_normalized.csv

Wejście: oczyszczony df z preprocessing.clean_data()
Wyjście: dwa pliki CSV gotowe do modelowania
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.settings import (
    CHARTS_STAGE5_DIR, 
    DATA_STAGE5_DIR, 
    DATA_DIR,
    BINARY_COLS, 
    ORDINAL_COLS, 
    ONEHOT_COLS, 
    SCALE_PREVIEW_COLS
)


# ---------------------------------------------------------------------------
# 1. Encoding
# ---------------------------------------------------------------------------

def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Koduje zmienne kategoryczne — binary, ordinal i one-hot."""
    df = df[df['Attrition'].notna()].copy()

    for col, mapping in BINARY_COLS.items():
        df[col] = df[col].map(mapping)

    for col, mapping in ORDINAL_COLS.items():
        df[col] = df[col].map(mapping)

    df = pd.get_dummies(df, columns=ONEHOT_COLS, drop_first=False, dtype=int)

    print(f'[stage5] Encoding zakończony — shape: {df.shape}')
    return df


# ---------------------------------------------------------------------------
# 2. Skalowanie
# ---------------------------------------------------------------------------

def scale(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Skaluje zmienne numeryczne — StandardScaler i MinMaxScaler.

    Pomijane: Attrition (zmienna docelowa), binarne 0/1 (już w zakresie [0,1]),
    one-hot 0/1. BusinessTravel (0/1/2) jest skalowany — bez tego wykraczałby
    poza zakres [0,1] w wersji znormalizowanej (MinMax).
    Zwraca dwa DataFrame: ze standaryzacją (Z-score) i normalizacją (Min-Max).
    """
    skip = (
        {'Attrition'}
        | set(BINARY_COLS.keys())
        | {c for c in df.columns if any(c.startswith(f'{base}_') for base in ONEHOT_COLS)}
    )
    num_cols = [c for c in df.select_dtypes(include='number').columns if c not in skip]

    df_std = df.copy()
    df_std[num_cols] = StandardScaler().fit_transform(df[num_cols])

    df_norm = df.copy()
    df_norm[num_cols] = MinMaxScaler().fit_transform(df[num_cols])

    print(f'[stage5] Skalowanie zakończone — {len(num_cols)} kolumn numerycznych')
    return df_std, df_norm


# ---------------------------------------------------------------------------
# 3. Wizualizacja przed/po skalowaniu
# ---------------------------------------------------------------------------

def plot_scaling_comparison(df_raw: pd.DataFrame, df_std: pd.DataFrame, df_norm: pd.DataFrame) -> None:
    """Histogramy top 4 kolumn numerycznych — przed skalowaniem, po standaryzacji i po normalizacji."""
    cols = [c for c in SCALE_PREVIEW_COLS if c in df_raw.columns]
    fig, axes = plt.subplots(len(cols), 3, figsize=(15, 4 * len(cols)))

    titles = ['Przed skalowaniem', 'Po standaryzacji (Z-score)', 'Po normalizacji (Min-Max)']
    dfs = [df_raw, df_std, df_norm]
    colors = ['#4c8cbf', '#e05c5c', '#5cba8a']

    for i, col in enumerate(cols):
        for j, (data, title, color) in enumerate(zip(dfs, titles, colors)):
            axes[i, j].hist(data[col].dropna(), bins=30, color=color, alpha=0.8, edgecolor='white')
            axes[i, j].set_title(f'{col} — {title}', fontsize=10)
            axes[i, j].set_xlabel(col)
            axes[i, j].set_ylabel('Liczba')

    plt.suptitle('Rozkłady zmiennych przed i po skalowaniu', fontsize=13)
    plt.tight_layout()
    plt.savefig(CHARTS_STAGE5_DIR / 'scaling_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[stage5] Zapisano: scaling_comparison.png')


# ---------------------------------------------------------------------------
# 4. Zapis
# ---------------------------------------------------------------------------

def save(df_std: pd.DataFrame, df_norm: pd.DataFrame) -> None:
    """Zapisuje finalne datasety do plików CSV."""
    DATA_STAGE5_DIR.mkdir(parents=True, exist_ok=True)

    path_std  = DATA_DIR / 'HR_model_standardized.csv'
    path_norm = DATA_DIR / 'HR_model_normalized.csv'

    df_std.to_csv(path_std, index=False)
    df_norm.to_csv(path_norm, index=False)

    print(f'[stage5] Zapisano: HR_model_standardized.csv  ({df_std.shape[0]} wierszy × {df_std.shape[1]} kolumn)')
    print(f'[stage5] Zapisano: HR_model_normalized.csv    ({df_norm.shape[0]} wierszy × {df_norm.shape[1]} kolumn)')


# ---------------------------------------------------------------------------
# Punkt wejścia
# ---------------------------------------------------------------------------

def run(df: pd.DataFrame) -> None:
    """Uruchamia preprocessing i zapisuje finalne datasety."""
    CHARTS_STAGE5_DIR.mkdir(parents=True, exist_ok=True)
    DATA_STAGE5_DIR.mkdir(parents=True, exist_ok=True)

    df_encoded = encode(df)
    df_std, df_norm = scale(df_encoded)
    plot_scaling_comparison(df_encoded, df_std, df_norm)
    save(df_std, df_norm)
