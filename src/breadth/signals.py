from __future__ import annotations

import pandas as pd


def classify_breadth_score(
    score: pd.Series,
    upper: float = 0.5,
    lower: float = -0.5,
) -> pd.Series:
    """
    Classify breadth regime using composite score thresholds.
    """
    regime = pd.Series("Neutral", index=score.index, dtype="object")

    regime[score > upper] = "Broadening"
    regime[score < lower] = "Narrowing"

    return regime


def classify_breadth_percentile(
    percentile: pd.Series,
    upper: float = 0.75,
    lower: float = 0.25,
) -> pd.Series:
    """
    Classify breadth regime using percentile thresholds.
    """
    regime = pd.Series("Neutral", index=percentile.index, dtype="object")

    regime[percentile >= upper] = "Broadening"
    regime[percentile <= lower] = "Narrowing"

    return regime


def add_index_divergence(
    breadth: pd.DataFrame,
    index_close: pd.Series,
    window: int = 20,
    score_col: str = "breadth_score",
) -> pd.DataFrame:
    """
    Add divergence flags between index returns and breadth score changes.
    """
    df = breadth.copy()

    if index_close is None:
        return df

    df["index_close"] = index_close.reindex(df.index)
    df[f"index_return_{window}d"] = df["index_close"].pct_change(window)
    df[f"breadth_score_change_{window}d"] = df[score_col].diff(window)

    df["bearish_divergence"] = (
        (df[f"index_return_{window}d"] > 0)
        & (df[f"breadth_score_change_{window}d"] < 0)
    )

    df["bullish_divergence"] = (
        (df[f"index_return_{window}d"] < 0)
        & (df[f"breadth_score_change_{window}d"] > 0)
    )

    return df