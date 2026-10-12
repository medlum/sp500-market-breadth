from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.breadth import config
from src.breadth.data_loader import get_sp500_tickers, load_price_frames
from src.breadth.universe import current_sp500_universe, select_universe_frames
from src.breadth.indicators import (
    advance_decline_breadth,
    mcclellan_style_oscillator,
    new_high_low_breadth,
    pct_above_moving_averages,
    rolling_positive_return_breadth,
    volume_breadth,
)
from src.breadth.composite import composite_breadth_score, rolling_percentile
from src.breadth.sectors import (
    build_sector_breadth,
    defensive_vs_cyclical_ratio,
)
from src.breadth.signals import (
    add_index_divergence,
    classify_breadth_percentile,
    classify_breadth_score,
)
from src.breadth.storage import ensure_directories, save_breadth


def build_breadth(
    frames: dict,
    universe: list[str],
    index_ticker: str | None,
) -> pd.DataFrame:
    """
    Build the full breadth table.
    """
    universe_frames = select_universe_frames(frames, universe)

    close = universe_frames["close"]
    high = universe_frames["high"]
    low = universe_frames["low"]
    volume = universe_frames["volume"]

    if close.empty:
        raise ValueError("No close data available for the S&P 500 universe.")

    ad = advance_decline_breadth(close)

    ma_breadth = pct_above_moving_averages(
        close,
        windows=config.MOVING_AVERAGE_WINDOWS,
    )

    return_breadth = rolling_positive_return_breadth(
        close,
        windows=config.RETURN_WINDOWS,
    )

    hl_breadth = new_high_low_breadth(
        high,
        low,
        windows=config.HIGH_LOW_WINDOWS,
    )

    vol_breadth = volume_breadth(close, volume)

    breadth = pd.concat(
        [
            ad,
            ma_breadth,
            return_breadth,
            hl_breadth,
            vol_breadth,
        ],
        axis=1,
    )

    breadth["mcclellan_style"] = mcclellan_style_oscillator(
        breadth["pct_net_advancers"]
    )

    new_low_cols = [
        col for col in breadth.columns if col.startswith("pct_new_low")
    ]

    for col in new_low_cols:
        breadth[f"neg_{col}"] = -breadth[col]

    score_columns = [
        col for col in config.SCORE_COLUMNS if col in breadth.columns
    ]

    breadth["breadth_score"] = composite_breadth_score(
        breadth,
        columns=score_columns,
        window=config.COMPOSITE_WINDOW,
    )

    breadth["breadth_score_percentile"] = rolling_percentile(
        breadth["breadth_score"],
        window=config.COMPOSITE_WINDOW,
    )

    breadth["breadth_score_20d_ma"] = (
        breadth["breadth_score"].rolling(20, min_periods=10).mean()
    )

    breadth["breadth_score_change_20d"] = breadth["breadth_score"].diff(20)

    breadth["breadth_regime"] = classify_breadth_score(
        breadth["breadth_score"]
    )

    breadth["breadth_regime_percentile"] = classify_breadth_percentile(
        breadth["breadth_score_percentile"]
    )

    if index_ticker and index_ticker in frames.get("close", pd.DataFrame()).columns:
        breadth = add_index_divergence(
            breadth,
            frames["close"][index_ticker],
            window=20,
        )

    return breadth


def build_and_save_sector_breadth(
    frames: dict,
    index_ticker: str | None,
) -> None:
    """Build per-sector indices/breadth and relative strength, then save them."""
    print("Building sector breadth and relative strength...")
    sector_index, breadth_long, rs = build_sector_breadth(
        frames=frames,
        index_ticker=index_ticker,
    )

    # Attach the defensive/cyclical ratio to the sector-index frame so the
    # dashboard has everything it needs in one file.
    sector_index["def_cyc_ratio"] = defensive_vs_cyclical_ratio(sector_index)

    sector_index.to_parquet(config.SECTOR_INDEX_OUTPUT)
    breadth_long.to_parquet(config.SECTOR_BREADTH_OUTPUT)

    print(f"Saved sector index  -> {config.SECTOR_INDEX_OUTPUT}")
    print(f"Saved sector breadth -> {config.SECTOR_BREADTH_OUTPUT}")
    print(
        f"  {len(breadth_long)} (date, sector) rows across "
        f"{breadth_long.index.get_level_values('sector').nunique()} sectors."
    )


def main() -> None:
    ensure_directories(
        config.DATA_DIR,
        config.RAW_DIR,
        config.BREADTH_DIR,
    )

    print("Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print(f"Loaded {len(tickers)} tickers.")

    print("Downloading/updating price data...")
    frames = load_price_frames(
        tickers=tickers,
        index_ticker=config.INDEX_TICKER,
        start=config.START_DATE,
        cache_dir=config.RAW_DIR,
    )

    universe = current_sp500_universe(
        tickers,
        exclude=[config.INDEX_TICKER],
    )

    print(f"Building breadth for {len(universe)} stocks...")

    breadth = build_breadth(
        frames=frames,
        universe=universe,
        index_ticker=config.INDEX_TICKER,
    )

    save_breadth(breadth, config.BREADTH_OUTPUT)

    print(f"Saved breadth output to: {config.BREADTH_OUTPUT}")
    print(breadth.tail(10))

    # Sector-level breadth + relative strength (used by the dashboard's
    # "Sectors" tab for rotation signals).
    try:
        build_and_save_sector_breadth(frames=frames, index_ticker=config.INDEX_TICKER)
    except Exception as exc:  # noqa: BLE001 - sector step is optional
        print(f"Skipping sector breadth (non-fatal): {exc}")


if __name__ == "__main__":
    main()