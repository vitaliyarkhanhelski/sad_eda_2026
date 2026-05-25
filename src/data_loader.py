"""
Wczytanie danych.
Wspólny dla całego zespołu — dostarcza surowe dane do pipeline'u.
"""

import pandas as pd
pd.set_option("display.max_columns", None)

from src.settings import DATA_DIR, DATASET_FILENAME

DATA_PATH = DATA_DIR / DATASET_FILENAME


def load_data() -> pd.DataFrame:
    """Wczytuje surowy dataset HR i zwraca DataFrame."""
    df = pd.read_csv(DATA_PATH)
    print(f"[data_loader] Wczytano: {df.shape[0]} wierszy, {df.shape[1]} kolumn")
    return df
