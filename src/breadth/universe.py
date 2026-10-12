from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pandas as pd


def current_sp500_universe(
    tickers: Iterable[str],
    exclude: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Return the stock universe used for breadth calculations.

    This MVP uses current constituents only.
    """
    exclude_set = set(exclude or [])
    return sorted(set(ticker for ticker in tickers if ticker not in exclude_set))


def select_universe_frames(
    frames: Dict[str, pd.DataFrame],
    universe: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Select only universe columns from price frames.
    """
    selected = {}

    for field, df in frames.items():
        cols = [col for col in universe if col in df.columns]
        selected[field] = df[cols].copy()

    return selected