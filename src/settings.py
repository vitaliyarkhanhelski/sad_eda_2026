"""Project constants and configuration."""

from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # __file__ is not defined in interactive sessions (e.g. IPython/REPL)
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
DATA_STAGE1_DIR = DATA_DIR / "stage1_preprocessing"
REPORTS_DIR = PROJECT_ROOT / "reports"
CHARTS_DIR = PROJECT_ROOT / "charts"
CHARTS_MSNO_DIR = CHARTS_DIR / "msno"
CHARTS_AGE_DIR = CHARTS_DIR / "age"
CHARTS_MONTHLY_INCOME_DIR = CHARTS_DIR / "monthly_income"
CHARTS_KNN_DIR = CHARTS_DIR / "knn"
DATASET_FILENAME = "HR.csv"
REPORT_FILENAME = "HR_profiling_report.html"

CONSTANT_COLS = ["EmployeeCount", "Over18", "StandardHours"]  # truly constant (single value)
COLS_TO_DROP = CONSTANT_COLS + ["EmployeeNumber"]              # all useless columns to remove
CONTINUOUS_COLS_FOR_OUTLIERS = ["MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
                                "YearsSinceLastPromotion", "NumCompaniesWorked"]
