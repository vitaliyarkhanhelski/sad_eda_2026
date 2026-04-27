"""Project constants and configuration."""

from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # __file__ is not defined in interactive sessions (e.g. IPython/REPL)
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATASET_FILENAME = "HR.csv"
REPORT_FILENAME = "HR_profiling_report.html"
