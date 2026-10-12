from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BREADTH_DIR = DATA_DIR / "breadth"

TICKER_CACHE = DATA_DIR / "sp500_tickers.csv"
BREADTH_OUTPUT = BREADTH_DIR / "sp500_breadth.parquet"

START_DATE = "2010-01-01"

# SPY is used as the index benchmark.
# You can also use "^GSPC", but SPY usually has cleaner volume data.
INDEX_TICKER = "SPY"

MOVING_AVERAGE_WINDOWS = (20, 50, 100, 200)
RETURN_WINDOWS = (5, 20, 63, 126, 252)
HIGH_LOW_WINDOWS = (20, 63, 252)

COMPOSITE_WINDOW = 252

SCORE_COLUMNS = [
    "pct_advancers",
    "pct_above_ma20",
    "pct_above_ma50",
    "pct_above_ma200",
    "pct_positive_5d",
    "pct_positive_20d",
    "pct_positive_63d",
    "pct_new_high_20d",
    "pct_new_high_252d",
    "pct_up_volume",
    "neg_pct_new_low_20d",
    "neg_pct_new_low_252d",
    "mcclellan_style",
]

# ---------------------------------------------------------------------------
# Sector rotation
# ---------------------------------------------------------------------------

SECTOR_INDEX_OUTPUT = BREADTH_DIR / "sp500_sector_index.parquet"
SECTOR_BREADTH_OUTPUT = BREADTH_DIR / "sp500_sector_breadth.parquet"

# Defensive (risk-off) vs cyclical (risk-on) groups, mirroring the common
# XLP+XLU+XLV vs XLY+XLI+XLB split.
DEFENSIVE_SECTORS = ("Consumer Staples", "Utilities", "Health Care")
CYCLICAL_SECTORS = ("Consumer Discretionary", "Industrials", "Materials")

# Relative-strength moving-average window (days the RS ratio must hold
# above/below to confirm a rotation).
SECTOR_RS_MA_WINDOW = 50