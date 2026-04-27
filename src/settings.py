"""Project constants and configuration."""

from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # __file__ is not defined in interactive sessions (e.g. IPython/REPL)
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
DATA_PREPROCESSING_DIR = DATA_DIR / "stage1_preprocessing"
REPORTS_DIR = PROJECT_ROOT / "reports"
CHARTS_DIR = PROJECT_ROOT / "charts"
CHARTS_PREPROCESSING_DIR = CHARTS_DIR / "stage1_preprocessing"
CHARTS_DESCRIPTIVE_DIR = CHARTS_DIR / "stage2_descriptive"
CHARTS_CORRELATION_DIR = CHARTS_DIR / "stage3_correlation"
CHARTS_PREPARATION_DIR = CHARTS_DIR / "stage4_preparation"
DATASET_FILENAME = "HR.csv"
REPORT_FILENAME = "HR_profiling_report.html"

CONSTANT_COLS = ["EmployeeCount", "Over18", "StandardHours"]  # truly constant (single value)
COLS_TO_DROP = CONSTANT_COLS + ["EmployeeNumber"]              # all useless columns to remove
CONTINUOUS_COLS_FOR_OUTLIERS = ["MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
                                "YearsSinceLastPromotion", "NumCompaniesWorked"]
