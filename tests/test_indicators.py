import pandas as pd

from src.breadth.indicators import advance_decline_breadth


def test_advance_decline_breadth():
    dates = pd.date_range("2024-01-01", periods=2, freq="D")

    close = pd.DataFrame(
        {
            "AAA": [100, 110],
            "BBB": [100, 90],
            "CCC": [100, 100],
        },
        index=dates,
    )

    ad = advance_decline_breadth(close)
    last_row = ad.iloc[-1]

    assert last_row["advancers"] == 1
    assert last_row["decliners"] == 1
    assert last_row["unchanged"] == 1
    assert last_row["valid_stocks"] == 3
    assert round(last_row["pct_advancers"], 4) == round(1 / 3, 4)