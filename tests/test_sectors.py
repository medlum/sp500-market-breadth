import numpy as np
import pandas as pd

from src.breadth import sectors


def _mapping():
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "sector": ["Technology", "Technology", "Energy", "Energy"],
        }
    )


def _frames():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    base = {"AAA": [10, 11, 12, 13, 14], "BBB": [20, 19, 21, 22, 23],
            "CCC": [30, 31, 30, 32, 33], "DDD": [40, 39, 38, 41, 42]}
    close = pd.DataFrame(base, index=dates)
    high = close + 1
    low = close - 1
    vol = pd.DataFrame(np.ones_like(close), index=dates, columns=close.columns)
    return {"close": close, "high": high, "low": low, "volume": vol}


def test_sector_equal_weight_index_starts_at_100():
    idx = sectors.sector_equal_weight_index(
        {"Technology": _frames()["close"][["AAA", "BBB"]]}
    )
    assert abs(idx["Technology"].iloc[0] - 100.0) < 1e-9
    assert idx["Technology"].iloc[-1] > 100.0


def test_rs_rank_strongest_is_one():
    rs = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [2.0, 1.0]}, index=pd.date_range("2024-01-01", periods=2)
    )
    ranks = sectors.rs_rank(rs)
    assert ranks["B"].iloc[0] == 1
    assert ranks["A"].iloc[1] == 1


def test_defensive_vs_cyclical_ratio():
    idx = pd.DataFrame(
        {"Utilities": [100.0, 110.0], "Industrials": [100.0, 90.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    ratio = sectors.defensive_vs_cyclical_ratio(
        pd.DataFrame(
            {"Consumer Staples": [100.0, 110.0], "Utilities": [100.0, 110.0],
             "Health Care": [100.0, 110.0], "Consumer Discretionary": [100.0, 90.0],
             "Industrials": [100.0, 90.0], "Materials": [100.0, 90.0]},
            index=idx.index,
        )
    )
    assert ratio.iloc[0] == 1.0
    assert ratio.iloc[-1] > 1.0  # defensives outperformed


def test_build_sector_breadth_with_monkeypatch(monkeypatch):
    monkeypatch.setattr(sectors, "load_sector_mapping", _mapping)
    sector_index, breadth_long, rs = sectors.build_sector_breadth(
        _frames(), index_ticker=None
    )
    assert set(sector_index.columns) == {"Energy", "Technology"}
    assert list(breadth_long.index.names) == ["date", "sector"]
    assert "Technology" in breadth_long.index.get_level_values("sector").unique()
    assert "rs_ratio" in breadth_long.columns
    assert not rs.empty
    # Regression: the dashboard filters by the 'date' level by name; make sure
    # that pattern works on freshly-built data.
    start = breadth_long.index.get_level_values("date").min()
    filtered = breadth_long[breadth_long.index.get_level_values("date") >= start]
    assert len(filtered) == len(breadth_long)


def test_rotation_summary_flags(monkeypatch):
    monkeypatch.setattr(sectors, "load_sector_mapping", _mapping)
    _, _, rs = sectors.build_sector_breadth(_frames(), index_ticker=None)
    ranks = sectors.rs_rank(rs)
    summary = sectors.rotation_summary(rs, ranks, lookback=1)
    assert set(summary.columns) >= {"sector", "current_rank", "signal"}
    assert len(summary) == rs.shape[1]