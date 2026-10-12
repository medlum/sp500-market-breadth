from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

from . import config

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
PRICE_FIELDS = ("close", "high", "low", "volume")


def get_sp500_tickers(refresh: bool = False) -> List[str]:
    """
    Get current S&P 500 tickers.

    This downloads from Wikipedia and caches the result locally.

    For serious historical research, replace this with a point-in-time
    S&P 500 membership database.
    """
    if config.TICKER_CACHE.exists() and not refresh:
        df = pd.read_csv(config.TICKER_CACHE)
        return df["ticker"].astype(str).str.strip().tolist()

    try:
        tables = pd.read_html(WIKI_SP500_URL)
        sp500 = tables[0]

        tickers = (
            sp500["Symbol"]
            .str.replace(".", "-", regex=False)
            .str.strip()
            .tolist()
        )

        tickers = sorted(set(tickers))

        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ticker": tickers}).to_csv(
            config.TICKER_CACHE,
            index=False,
        )

        return tickers

    except Exception as exc:
        raise RuntimeError(
            "Could not download S&P 500 tickers from Wikipedia. "
            f"Create a CSV file at {config.TICKER_CACHE} "
            "with a column named 'ticker'."
        ) from exc


def _chunks(items: List[str], n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize price DataFrame index to timezone-naive DatetimeIndex.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    return df


def _xs_field(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    """
    Extract a field from yfinance MultiIndex columns.
    """
    if field in raw.columns.get_level_values(0):
        return raw.xs(field, axis=1, level=0)

    return pd.DataFrame(index=raw.index)


def _extract_frames(raw: pd.DataFrame, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Convert raw yfinance download into separate close, high, low, volume frames.
    """
    if raw is None or raw.empty:
        return {field: pd.DataFrame() for field in PRICE_FIELDS}

    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(0).unique()
        close_field = "Adj Close" if "Adj Close" in fields else "Close"

        return {
            "close": _xs_field(raw, close_field),
            "high": _xs_field(raw, "High"),
            "low": _xs_field(raw, "Low"),
            "volume": _xs_field(raw, "Volume"),
        }

    if len(tickers) != 1:
        raise RuntimeError(
            "Unexpected yfinance data format. "
            "Expected MultiIndex columns for multiple tickers."
        )

    ticker = tickers[0]
    close_col = "Adj Close" if "Adj Close" in raw.columns else "Close"

    close = raw[[close_col]].copy()
    close.columns = [ticker]

    high = raw[["High"]].copy()
    high.columns = [ticker]

    low = raw[["Low"]].copy()
    low.columns = [ticker]

    volume = raw[["Volume"]].copy()
    volume.columns = [ticker]

    return {
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
    }


def _download_frames(tickers: List[str], start: str) -> Dict[str, pd.DataFrame]:
    """
    Download price frames for all tickers in chunks.
    """
    collected = {field: [] for field in PRICE_FIELDS}

    for chunk in _chunks(tickers, 50):
        raw = yf.download(
            tickers=chunk,
            start=start,
            auto_adjust=False,
            group_by="column",
            progress=False,
            threads=True,
        )

        extracted = _extract_frames(raw, chunk)

        for field in PRICE_FIELDS:
            df = extracted.get(field)
            if df is not None and not df.empty:
                collected[field].append(df)

    out = {}

    for field in PRICE_FIELDS:
        if collected[field]:
            df = pd.concat(collected[field], axis=1)
            df = df.loc[:, ~df.columns.duplicated()]
        else:
            df = pd.DataFrame()

        df = _normalize_index(df)

        if not df.empty:
            df.sort_index(axis=1, inplace=True)

        out[field] = df

    return out


def _merge_frames(
    old: Dict[str, pd.DataFrame],
    new: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Merge newly downloaded price frames into cached frames.
    """
    merged = {}

    for field in PRICE_FIELDS:
        old_df = old.get(field, pd.DataFrame())
        new_df = new.get(field, pd.DataFrame())

        if new_df.empty:
            merged[field] = old_df.copy()
            continue

        if old_df.empty:
            merged[field] = new_df.copy()
            continue

        combined = pd.concat([old_df, new_df], axis=0)
        combined = _normalize_index(combined)
        combined.sort_index(axis=1, inplace=True)

        merged[field] = combined

    return merged


def load_price_frames(
    tickers: List[str],
    index_ticker: str | None = None,
    start: str | None = None,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Load price frames from cache, updating if necessary.

    Returns:
        {
            "close": DataFrame,
            "high": DataFrame,
            "low": DataFrame,
            "volume": DataFrame,
        }
    """
    start = start or config.START_DATE
    cache_dir = Path(cache_dir) if cache_dir is not None else config.RAW_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_tickers = sorted(set(tickers + ([index_ticker] if index_ticker else [])))

    paths = {field: cache_dir / f"{field}.parquet" for field in PRICE_FIELDS}

    if not refresh and all(path.exists() for path in paths.values()):
        cached = {
            field: _normalize_index(pd.read_parquet(paths[field]))
            for field, path in paths.items()
        }

        close = cached.get("close")

        if close is not None and not close.empty:
            last_date = close.index.max()
            today = pd.Timestamp.today().normalize()

            if pd.notna(last_date) and last_date >= today:
                return cached

            # Overlap by 7 days to reduce the chance of missing corrections.
            update_start = max(
                last_date - pd.Timedelta(days=7),
                pd.Timestamp(start),
            )

            new_data = _download_frames(
                all_tickers,
                update_start.strftime("%Y-%m-%d"),
            )

            updated = _merge_frames(cached, new_data)

            for field in PRICE_FIELDS:
                updated[field].to_parquet(paths[field])

            return updated

    data = _download_frames(all_tickers, start)

    for field in PRICE_FIELDS:
        data[field].to_parquet(paths[field])

    return data