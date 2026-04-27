"""
Wczytanie danych i generowanie wstępnego raportu profilującego.
Wspólny dla całego zespołu — dostarcza surowe dane do pipeline'u.
"""

import pandas as pd
pd.set_option("display.max_columns", None)

from ydata_profiling import ProfileReport

from src.settings import DATA_DIR, REPORTS_DIR, DATASET_FILENAME, REPORT_FILENAME

DATA_PATH = DATA_DIR / DATASET_FILENAME
REPORT_PATH = REPORTS_DIR / REPORT_FILENAME


def load_data() -> pd.DataFrame:
    """Wczytuje surowy dataset HR i zwraca DataFrame."""
    df = pd.read_csv(DATA_PATH)
    print(f"[data_loader] Wczytano: {df.shape[0]} wierszy, {df.shape[1]} kolumn")
    return df


def generate_report(df: pd.DataFrame) -> None:
    """Generuje automatyczny raport EDA (ydata_profiling) do folderu report/."""
    print("[data_loader] Generowanie raportu profilującego...")
    profile = ProfileReport(
        df,
        title="HR Dataset — Raport profilujący",
        explorative=True,
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": True},
        }
    )
    profile.to_file(REPORT_PATH)
    print(f"[data_loader] Raport zapisany: {REPORT_PATH}")
