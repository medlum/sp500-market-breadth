from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_directories(*directories):
    """
    Create directories if they do not exist.
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def save_breadth(breadth: pd.DataFrame, path: Path | str) -> None:
    """
    Save breadth DataFrame to Parquet.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    breadth.to_parquet(path)


def load_breadth(path: Path | str) -> pd.DataFrame:
    """
    Load breadth DataFrame from Parquet.
    """
    return pd.read_parquet(path)