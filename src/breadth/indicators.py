from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _safe_pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Compute numerator / denominator, replacing division by zero with NaN.
    """
    return numerator / denominator.replace(0, np.nan)


def advance_decline_breadth(close: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily advance/decline breadth.
    """
    returns = close.pct_change()

    valid = returns.notna().sum(axis=1)
    advancers = ((returns > 0) & returns.notna()).sum(axis=1)
    decliners = ((returns < 0) & returns.notna()).sum(axis=1)
    unchanged = valid - advancers - decliners

    ad = pd.DataFrame(
        {
            "valid_stocks": valid,
            "advancers": advancers,
            "decliners": decliners,
            "unchanged": unchanged,
            "net_advancers": advancers - decliners,
        }
    )

    ad["pct_advancers"] = _safe_pct(ad["advancers"], ad["valid_stocks"])
    ad["pct_decliners"] = _safe_pct(ad["decliners"], ad["valid_stocks"])
    ad["pct_net_advancers"] = _safe_pct(ad["net_advancers"], ad["valid_stocks"])

    ad["ad_ratio"] = ad["advancers"] / ad["decliners"].replace(0, np.nan)
    ad["ad_ratio"] = ad["ad_ratio"].replace([np.inf, -np.inf], np.nan)

    ad["ad_line"] = ad["net_advancers"].cumsum()
    ad["normalized_ad_line"] = ad["pct_net_advancers"].cumsum()

    return ad


def pct_above_moving_averages(
    close: pd.DataFrame,
    windows: Iterable[int] = (20, 50, 100, 200),
) -> pd.DataFrame:
    """
    Compute percentage of stocks above key moving averages.
    """
    out = {}

    for window in windows:
        min_periods = max(1, int(window * 0.8))
        ma = close.rolling(window=window, min_periods=min_periods).mean()

        valid = close.notna() & ma.notna()
        above = (close > ma) & valid

        out[f"pct_above_ma{window}"] = _safe_pct(
            above.sum(axis=1),
            valid.sum(axis=1),
        )

    return pd.DataFrame(out)


def rolling_positive_return_breadth(
    close: pd.DataFrame,
    windows: Iterable[int] = (5, 20, 63, 126, 252),
) -> pd.DataFrame:
    """
    Compute percentage of stocks with positive returns over lookback windows.
    """
    out = {}

    for window in windows:
        returns = close.pct_change(window)

        valid = returns.notna()
        positive = (returns > 0) & valid

        out[f"pct_positive_{window}d"] = _safe_pct(
            positive.sum(axis=1),
            valid.sum(axis=1),
        )

    return pd.DataFrame(out)


def new_high_low_breadth(
    high: pd.DataFrame,
    low: pd.DataFrame,
    windows: Iterable[int] = (20, 63, 252),
) -> pd.DataFrame:
    """
    Compute percentage of stocks making rolling new highs and new lows.
    """
    out = {}

    for window in windows:
        min_periods = max(1, int(window * 0.8))

        rolling_high = high.rolling(window=window, min_periods=min_periods).max()
        rolling_low = low.rolling(window=window, min_periods=min_periods).min()

        valid_high = high.notna() & rolling_high.notna()
        valid_low = low.notna() & rolling_low.notna()

        new_high = (high >= rolling_high) & valid_high
        new_low = (low <= rolling_low) & valid_low

        out[f"pct_new_high_{window}d"] = _safe_pct(
            new_high.sum(axis=1),
            valid_high.sum(axis=1),
        )

        out[f"pct_new_low_{window}d"] = _safe_pct(
            new_low.sum(axis=1),
            valid_low.sum(axis=1),
        )

    return pd.DataFrame(out)


def volume_breadth(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """
    Compute up-volume and down-volume breadth.
    """
    returns = close.pct_change()
    volume_filled = volume.fillna(0.0)

    up_volume = volume_filled.where(returns > 0, 0.0).sum(axis=1)
    down_volume = volume_filled.where(returns < 0, 0.0).sum(axis=1)
    total_volume = volume_filled.sum(axis=1)

    out = pd.DataFrame(
        {
            "up_volume": up_volume,
            "down_volume": down_volume,
            "total_volume": total_volume,
        }
    )

    out["up_down_volume_ratio"] = out["up_volume"] / out["down_volume"].replace(
        0,
        np.nan,
    )
    out["up_down_volume_ratio"] = out["up_down_volume_ratio"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    out["pct_up_volume"] = _safe_pct(out["up_volume"], total_volume)

    return out


def mcclellan_style_oscillator(
    pct_net_advancers: pd.Series,
    short_span: int = 19,
    long_span: int = 39,
) -> pd.Series:
    """
    McClellan-style oscillator based on normalized net advancers.
    """
    short_ema = pct_net_advancers.ewm(span=short_span, adjust=False).mean()
    long_ema = pct_net_advancers.ewm(span=long_span, adjust=False).mean()

    return short_ema - long_ema