from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config
from .indicators import (
    advance_decline_breadth,
    new_high_low_breadth,
    pct_above_moving_averages,
    rolling_positive_return_breadth,
    volume_breadth,
)

# Classic risk-on / risk-off splits (matches the common XLP+XLU+XLV vs XLY+XLI+XLB).
DEFENSIVE_SECTORS = list(config.DEFENSIVE_SECTORS)
CYCLICAL_SECTORS = list(config.CYCLICAL_SECTORS)


def load_sector_mapping(path: Optional[object] = None) -> pd.DataFrame:
    """Load the ticker -> GICS sector mapping from the cached ticker file.

    The Wikipedia scrape already stores a 'gics sector' column in
    ``data/sp500_tickers.csv``; this just surfaces it.
    """
    path = Path(path) if path is not None else config.TICKER_CACHE
    df = pd.read_csv(path)

    # Be tolerant of column-name casing/spacing across cache versions.
    cols = {str(c).strip().lower(): c for c in df.columns}
    ticker_col = cols.get("ticker", cols.get("symbol"))
    sector_col = cols.get("gics sector", cols.get("sector"))

    if ticker_col is None or sector_col is None:
        raise ValueError(
            f"Ticker cache at {path} is missing 'ticker'/'gics sector' columns. "
            "Re-run the pipeline to refresh it."
        )

    out = df[[ticker_col, sector_col]].copy()
    out.columns = ["ticker", "sector"]
    out = out.dropna(subset=["sector"])
    out["ticker"] = out["ticker"].astype(str).str.strip()
    out["sector"] = out["sector"].astype(str).str.strip()
    return out


def ticker_to_sector(mapping: pd.DataFrame) -> Dict[str, str]:
    """Return a {ticker: sector} dict from a mapping DataFrame."""
    return dict(zip(mapping["ticker"], mapping["sector"]))


def group_frames_by_sector(
    frames: Dict[str, pd.DataFrame],
    mapping: pd.DataFrame,
    exclude: Optional[List[str]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Slice each price field into per-sector frames.

    Returns {sector: {"close": df, "high": df, "low": df, "volume": df}}.
    Tickers with no sector (e.g. the index) are dropped.
    """
    t2s = ticker_to_sector(mapping)
    exclude_set = set(exclude or [])
    selected = {t: s for t, s in t2s.items() if t not in exclude_set}

    sectors: Dict[str, Dict[str, pd.DataFrame]] = {}
    for field, df in frames.items():
        if df is None or df.empty:
            continue
        for ticker, sector in selected.items():
            if ticker not in df.columns:
                continue
            sectors.setdefault(sector, {}).setdefault(field, pd.DataFrame(index=df.index))
            sectors[sector][field][ticker] = df[ticker]
    return sectors


def sector_equal_weight_index(close_by_sector: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build an equal-weight price index per sector.

    Each stock is rebased to 100 at its first valid observation, then the
    cross-sectional mean (skipna) is taken per day, so the index starts at 100
    and grows with the (equal-weighted) average constituent.
    """
    out = {}
    for sector, fr in close_by_sector.items():
        if fr is None or fr.empty:
            out[sector] = pd.Series(np.nan, index=fr.index, dtype=float)
            continue
        firsts = fr.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
        norm = fr.div(firsts) * 100.0
        out[sector] = norm.mean(axis=1, skipna=True)
    return pd.DataFrame(out).sort_index(axis=1)


def sector_relative_strength(
    sector_index: pd.DataFrame,
    index_close: pd.Series,
) -> pd.DataFrame:
    """Relative-strength ratio = sector index / benchmark, per sector per day.

    A rising ratio means the sector is outperforming the benchmark (SPY).
    """
    aligned = index_close.reindex(sector_index.index)
    return sector_index.div(aligned, axis=0)


def rs_rank(rs: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank of RS across sectors each day (1 = strongest)."""
    return rs.rank(axis=1, ascending=False, method="min")


def defensive_vs_cyclical_ratio(sector_index: pd.DataFrame) -> pd.Series:
    """Defensive sectors index / Cyclical sectors index.

    Rising  => capital rotating into defensives (risk-off).
    Falling => capital rotating into cyclicals (risk-on).
    """
    def_cols = [c for c in DEFENSIVE_SECTORS if c in sector_index.columns]
    cyc_cols = [c for c in CYCLICAL_SECTORS if c in sector_index.columns]
    if not def_cols or not cyc_cols:
        return pd.Series(np.nan, index=sector_index.index, name="def_cyc_ratio")
    def_avg = sector_index[def_cols].mean(axis=1, skipna=True)
    cyc_avg = sector_index[cyc_cols].mean(axis=1, skipna=True)
    return (def_avg / cyc_avg).rename("def_cyc_ratio")


def _sector_breadth_one(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the focused per-sector breadth metrics for one sector."""
    parts = []
    if not close.empty:
        ad = advance_decline_breadth(close)
        parts.append(ad[["pct_advancers"]])
        parts.append(
            pct_above_moving_averages(close, config.MOVING_AVERAGE_WINDOWS)[
                ["pct_above_ma20", "pct_above_ma50", "pct_above_ma200"]
            ]
        )
        parts.append(
            rolling_positive_return_breadth(close, config.RETURN_WINDOWS)[
                ["pct_positive_20d", "pct_positive_63d"]
            ]
        )
    if not high.empty and not low.empty:
        parts.append(
            new_high_low_breadth(high, low, config.HIGH_LOW_WINDOWS)[
                ["pct_new_high_252d", "pct_new_low_252d"]
            ]
        )
    if not close.empty and not volume.empty:
        parts.append(volume_breadth(close, volume)[["pct_up_volume"]])
    if not parts:
        return pd.DataFrame(index=close.index)
    return pd.concat(parts, axis=1)


def build_sector_breadth(
    frames: Dict[str, pd.DataFrame],
    index_ticker: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build sector indices, per-sector breadth, and relative strength.

    Returns
    -------
    sector_index : DataFrame
        date x sectors (equal-weight indices) plus the benchmark column.
    breadth_long : DataFrame
        long format, MultiIndex (date, sector) with per-sector breadth +
        ``rs_ratio`` and ``rs_rank``.
    rs : DataFrame
        date x sectors relative-strength ratios (handy for charts/flags).
    """
    mapping = load_sector_mapping()
    sector_frames = group_frames_by_sector(frames, mapping, exclude=[index_ticker])

    close_by_sector = {s: fr.get("close", pd.DataFrame()) for s, fr in sector_frames.items()}
    sector_index = sector_equal_weight_index(close_by_sector)

    if index_ticker and index_ticker in frames.get("close", pd.DataFrame()).columns:
        sector_index[index_ticker] = frames["close"][index_ticker]

    drop_cols = [index_ticker] if index_ticker and index_ticker in sector_index.columns else []
    bench = (
        sector_index[index_ticker]
        if index_ticker and index_ticker in sector_index.columns
        else pd.Series(np.nan, index=sector_index.index)
    )
    rs = sector_relative_strength(sector_index.drop(columns=drop_cols), bench)
    ranks = rs_rank(rs)

    records: List[pd.DataFrame] = []
    for sector, fr in sector_frames.items():
        bdf = _sector_breadth_one(
            fr.get("close", pd.DataFrame()),
            fr.get("high", pd.DataFrame()),
            fr.get("low", pd.DataFrame()),
            fr.get("volume", pd.DataFrame()),
        )
        if bdf.empty:
            continue
        bdf = bdf.copy()
        bdf["sector"] = sector
        bdf["rs_ratio"] = rs[sector] if sector in rs.columns else np.nan
        bdf["rs_rank"] = ranks[sector] if sector in ranks.columns else np.nan
        records.append(bdf)

    if records:
        breadth_long = pd.concat(records)
        breadth_long = breadth_long.set_index("sector", append=True).sort_index()
        # Normalise level names: the inherited first level is named after the
        # raw price index (often 'Date'), but the rest of the code and the
        # dashboard expect ['date', 'sector'].
        breadth_long.index = breadth_long.index.set_names(["date", "sector"])
    else:
        breadth_long = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["date", "sector"])
        )

    return sector_index, breadth_long, rs


def rotation_summary(
    rs: pd.DataFrame,
    ranks: pd.DataFrame,
    lookback: int = 63,
    rs_ma_window: Optional[int] = None,
) -> pd.DataFrame:
    """Summarise recent sector rotation.

    For each sector, returns its current RS rank, the rank ``lookback`` days
    ago, whether RS is above its moving average, and a plain-English flag.
    """
    rs_ma_window = rs_ma_window or config.SECTOR_RS_MA_WINDOW
    rs_ma = rs.rolling(rs_ma_window, min_periods=max(5, rs_ma_window // 2)).mean()
    n = len(ranks)
    rows = []
    for sector in rs.columns:
        cur_rank = float(ranks[sector].iloc[-1]) if n else np.nan
        prev_rank = (
            float(ranks[sector].iloc[-lookback - 1])
            if n > lookback and pd.notna(ranks[sector].iloc[-lookback - 1])
            else np.nan
        )
        cur_rs = float(rs[sector].iloc[-1]) if n else np.nan
        cur_rs_ma = float(rs_ma[sector].iloc[-1]) if n else np.nan

        delta = (prev_rank - cur_rank) if pd.notna(prev_rank) and pd.notna(cur_rank) else np.nan
        if pd.isna(delta):
            flag = "\u2014"
        elif delta >= 3:
            flag = "Rotating IN (leadership improving)"
        elif delta <= -3:
            flag = "Rotating OUT (leadership fading)"
        else:
            flag = "Stable"

        above_ma = bool(pd.notna(cur_rs) and pd.notna(cur_rs_ma) and cur_rs > cur_rs_ma)

        rows.append(
            {
                "sector": sector,
                "current_rank": cur_rank,
                f"rank_{lookback}d_ago": prev_rank,
                "rank_change": -delta if pd.notna(delta) else np.nan,
                "rs_vs_ma50": "Above" if above_ma else "Below",
                "signal": flag,
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("current_rank").reset_index(drop=True)
    return summary