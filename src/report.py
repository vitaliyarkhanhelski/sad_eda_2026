"""
Generowanie automatycznego raportu profilującego (ydata_profiling).
"""

import pandas as pd
from ydata_profiling import ProfileReport

from src.settings import REPORTS_DIR, REPORT_FILENAME

REPORT_PATH = REPORTS_DIR / REPORT_FILENAME


def generate_report(df: pd.DataFrame) -> None:
    """Generuje automatyczny raport EDA (ydata_profiling) do folderu reports/."""
    print("[report] Generowanie raportu profilującego...")
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
    print(f"[report] Raport zapisany: {REPORT_PATH}")
