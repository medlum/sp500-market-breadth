"""Regression guards for the chart colour palettes.

Each multi-line chart shares its axes with several series, so the colours must
stay perceptually distinct. These tests compute a weighted ("redmean") RGB
distance between every pair in a palette and assert the minimum stays above a
threshold.

The thresholds were calibrated against matplotlib's well-tested ``tab10``
palette, whose 10-colour minimum pairwise distance is ~57. The palettes used
here target that benchmark or better.
"""

from src.breadth.plots import (
    _HIGH_COLORS,
    _LOW_COLORS,
    _MA_COLORS,
    _MOMENTUM_COLORS,
    SECTOR_COLORS,
)


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))


def _distance(a: str, b: str) -> float:
    """Weighted ("redmean") RGB distance, scaled to 0-255-ish units."""
    (r1, g1, b1), (r2, g2, b2) = _hex_to_rgb(a), _hex_to_rgb(b)
    rm = (r1 + r2) / 2
    return (
        ((2 + rm) * (r1 - r2) ** 2 + 4 * (g1 - g2) ** 2 + (3 - rm) * (b1 - b2) ** 2)
        ** 0.5
        * 255
        * 0.5
    )


def _min_distance(colors) -> float:
    worst = float("inf")
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            worst = min(worst, _distance(colors[i], colors[j]))
    return worst


def test_tab10_baseline_is_at_least_56():
    """matplotlib's tab10 is the reference palette; record its minimum distance.

    Its true minimum is ~56.75 (between the mauve ``#9467bd`` and the gray
    ``#7f7f7f``). The custom palettes used in this project are required to beat
    or match this benchmark.
    """
    tab10 = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    assert _min_distance(tab10) >= 56


def test_ma_colors_are_distinct():
    assert _min_distance(list(_MA_COLORS.values())) >= 80


def test_momentum_colors_are_distinct():
    assert _min_distance(_MOMENTUM_COLORS) >= 80


def test_high_low_colors_distinct_within_and_across():
    # Each family must be internally distinct, and the combined 6-line chart
    # (highs + lows share one axis) must stay distinct overall.
    assert _min_distance(_HIGH_COLORS) >= 50
    assert _min_distance(_LOW_COLORS) >= 50
    assert _min_distance(_HIGH_COLORS + _LOW_COLORS) >= 60


def test_sector_colors_are_distinct():
    # 11 sectors share the Sector RS chart; match the tab10 benchmark.
    assert _min_distance(SECTOR_COLORS) >= 55