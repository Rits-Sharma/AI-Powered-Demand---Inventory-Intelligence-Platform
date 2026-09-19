from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

for path in [PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

SALES_FILE = RAW_DIR / "sales_daily.csv"
SKU_FILE = RAW_DIR / "sku_master.csv"
CALENDAR_FILE = RAW_DIR / "calendar.csv"
CALENDAR_ALT_FILE = RAW_DIR / "calender.csv"
INVENTORY_FILE = RAW_DIR / "inventory_snapshots.csv"
