from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Compute rolling z-score.
    """
    min_periods = max(20, int(window * 0.7))

    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)

    return (series - mean) / std.replace(0, np.nan)


def rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Compute rolling percentile rank.
    """
    min_periods = max(20, int(window * 0.7))

    return series.rolling(window=window, min_periods=min_periods).rank(pct=True)


def composite_breadth_score(
    breadth: pd.DataFrame,
    columns: List[str],
    window: int = 252,
) -> pd.Series:
    """
    Create a composite breadth score from multiple breadth indicators.

    Each indicator is converted to a rolling z-score, then averaged.
    """
    available = [col for col in columns if col in breadth.columns]

    if not available:
        return pd.Series(np.nan, index=breadth.index, name="breadth_score")

    zscores = pd.DataFrame(index=breadth.index)

    for col in available:
        zscores[col] = rolling_zscore(breadth[col], window=window)

    return zscores.mean(axis=1)