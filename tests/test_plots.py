import pandas as pd
import plotly.graph_objects as go

from src.breadth.plots import (
    plot_ad_line_vs_index,
    plot_advance_decline,
    plot_breadth_score,
    plot_index_vs_breadth,
    plot_mcclellan,
    plot_momentum,
    plot_new_high_low,
    plot_percent_above_moving_averages,
    plot_regime_timeline,
    plot_volume_breadth,
    _apply_layout,
    _contiguous_runs,
    _date_axis_ticks,
    _label,
    _rebase,
)


def _sample_breadth():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    return pd.DataFrame(
        {
            "breadth_score": [0.1] * 60,
            "breadth_score_percentile": [0.5] * 60,
            "breadth_score_20d_ma": [0.1] * 60,
            "breadth_regime": ["Neutral"] * 30 + ["Broadening"] * 30,
            "pct_above_ma20": [0.6] * 60,
            "pct_above_ma50": [0.5] * 60,
            "pct_above_ma100": [0.45] * 60,
            "pct_above_ma200": [0.4] * 60,
            "pct_positive_5d": [0.5] * 60,
            "pct_positive_20d": [0.5] * 60,
            "pct_positive_63d": [0.5] * 60,
            "pct_positive_126d": [0.5] * 60,
            "pct_positive_252d": [0.5] * 60,
            "pct_new_high_20d": [0.1] * 60,
            "pct_new_low_20d": [0.05] * 60,
            "pct_new_high_252d": [0.03] * 60,
            "pct_new_low_252d": [0.01] * 60,
            "pct_up_volume": [0.55] * 60,
            "mcclellan_style": [0.0] * 60,
            "ad_line": list(range(60)),
            "index_close": list(range(100, 160)),
            "net_advancers": [5] * 60,
            "bearish_divergence": [False] * 30 + [True] * 30,
            "bullish_divergence": [False] * 60,
        },
        index=dates,
    )


ALL_PLOT_FUNCS = [
    plot_breadth_score,
    plot_percent_above_moving_averages,
    plot_advance_decline,
    plot_new_high_low,
    plot_volume_breadth,
    plot_index_vs_breadth,
    plot_ad_line_vs_index,
    plot_mcclellan,
    plot_regime_timeline,
    plot_momentum,
]


def test_all_plots_return_figures_with_traces():
    df = _sample_breadth()
    for fn in ALL_PLOT_FUNCS:
        fig = fn(df)
        assert isinstance(fig, go.Figure), f"{fn.__name__} did not return a Figure"
        assert len(fig.data) >= 1, f"{fn.__name__} produced no traces"


def test_empty_inputs_fall_back_gracefully():
    empty = pd.DataFrame(index=pd.date_range("2024-01-01", periods=2, freq="D"))
    for fn in ALL_PLOT_FUNCS:
        fig = fn(empty)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0, f"{fn.__name__} should return an empty fallback"


def test_label_mapping_is_human_readable():
    assert _label("pct_above_ma50") == "% > 50-Day MA"
    assert _label("pct_new_low_252d") == "New 252-D Lows"


def test_contiguous_runs_finds_two_runs():
    s = pd.Series(
        [False] * 5 + [True] * 3 + [False] * 2 + [True] * 2,
        index=pd.date_range("2024-01-01", periods=12, freq="D"),
    )
    assert len(_contiguous_runs(s.astype(bool))) == 2


def test_rebase_computes_percent_change():
    s = pd.Series([100, 110, 105], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    r = _rebase(s)
    assert abs(r.iloc[0]) < 1e-9
    assert abs(r.iloc[1] - 10.0) < 1e-9


# ---------------------------------------------------------------------------
# Date-axis tick de-duplication (regression for duplicate "Aug 2026" labels)
# ---------------------------------------------------------------------------

def test_date_axis_ticks_3m_uses_day_format():
    """A 3-month window uses day-level labels anchored to the first data date
    (so ticks don't drift onto weekends)."""
    dates = pd.date_range("2026-07-01", periods=63, freq="D")
    fig = go.Figure(go.Scatter(x=dates, y=list(range(63))))
    dtick, fmt, tick0 = _date_axis_ticks(fig)
    assert fmt == "%b %d"
    assert dtick is not None
    assert tick0 == "2026-07-01"


def test_date_axis_ticks_1y_uses_monthly():
    dates = pd.date_range("2025-01-01", periods=365, freq="D")
    fig = go.Figure(go.Scatter(x=dates, y=list(range(365))))
    dtick, fmt, tick0 = _date_axis_ticks(fig)
    assert dtick == "M1"
    assert fmt == "%b %Y"
    assert tick0 is None


def test_date_axis_ticks_all_history_uses_yearly():
    dates = pd.date_range("2010-01-01", periods=16 * 365, freq="D")
    fig = go.Figure(go.Scatter(x=dates, y=list(range(len(dates)))))
    dtick, fmt, tick0 = _date_axis_ticks(fig)
    assert dtick == "M12"
    assert fmt == "%Y"
    assert tick0 is None


def test_date_axis_labels_never_duplicate():
    """The bug: 3-month windows showed two identical 'Aug 2026' axis labels.
    With the new step+format, consecutive ticks are far enough apart that
    formatted labels are always unique. Ticks are stepped from the returned
    tick0, which is the anchor Plotly actually uses for placement."""
    dates = pd.date_range("2026-07-01", periods=63, freq="D")
    fig = go.Figure(go.Scatter(x=dates, y=list(range(63))))
    dtick, fmt, tick0 = _date_axis_ticks(fig)

    if isinstance(dtick, str):
        step = pd.DateOffset(months=int(dtick[1:]))
    else:
        step = pd.Timedelta(milliseconds=dtick)

    start = pd.Timestamp(tick0) if tick0 is not None else dates[0]
    ticks = []
    d = start
    while d <= dates[-1]:
        ticks.append(d)
        d = d + step
    labels = [t.strftime(fmt) for t in ticks]
    assert len(labels) == len(set(labels)), f"Duplicate labels: {labels}"


def test_apply_layout_sets_dtick_on_short_range():
    dates = pd.date_range("2026-07-01", periods=63, freq="D")
    fig = go.Figure(go.Scatter(x=dates, y=list(range(63))))
    _apply_layout(fig, "Test", "Y")
    assert fig.layout.xaxis.dtick is not None
    assert fig.layout.xaxis.tick0 is not None
    assert fig.layout.xaxis.tick0 == "2026-07-01"


def test_short_range_tick0_anchors_ticks_to_real_weekdays():
    """Regression for the '3 Months' axis bug: without an explicit tick0 Plotly
    anchors date-axis ticks at 2000-01-01 (a Saturday), so a 14-day numeric
    step lands every tick on a weekend that exists nowhere in the data. The
    fix anchors tick0 at the first data date, keeping ticks on real weekdays
    that are present in the series."""
    # 2026-07-13 is a Monday.
    dates = pd.date_range("2026-07-13", periods=63, freq="D")
    fig = go.Figure(go.Scatter(x=dates, y=list(range(63))))
    dtick, fmt, tick0 = _date_axis_ticks(fig)

    assert dtick is not None and not isinstance(dtick, str)
    assert fmt == "%b %d"
    assert tick0 == "2026-07-13"

    anchor = pd.Timestamp(tick0)
    assert anchor.weekday() < 5, "anchor should be a Mon-Fri trading day"

    # Simulate Plotly tick placement: tick0 + n*dtick within the data range.
    step = pd.Timedelta(milliseconds=dtick)
    ticks = []
    d = anchor
    while d <= dates[-1]:
        if d >= dates[0]:
            ticks.append(d)
        d = d + step
    assert ticks, "expected at least one tick in range"
    # 14 days == 2 weeks, so every tick shares the anchor's weekday and never
    # drifts onto a weekend.
    for t in ticks:
        assert t.weekday() == anchor.weekday()
    # And every tick is a date that is actually present in the series.
    for t in ticks:
        assert t in dates


def test_apply_layout_skips_dtick_for_non_date_axis():
    """Non-date axes (e.g. the sector matrix) must not get a date dtick."""
    fig = go.Figure(go.Heatmap(z=[[1, 2], [3, 4]], x=["A", "B"], y=["X", "Y"]))
    _apply_layout(fig, "Test", "Y", date_axis=False)
    assert fig.layout.xaxis.dtick is None
