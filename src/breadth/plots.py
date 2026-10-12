from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = {
    "bull": "#2eab5a",
    "bear": "#d64545",
    "neutral": "#9aa3ad",
    "accent": "#1f6feb",
    "index": "#f59e0b",
}

# Distinct hues for each lookback so lines in the same chart are easy to tell
# apart. (Previously all four MA windows were shades of blue, which made the
# 20/50/100/200-day lines hard to distinguish.)
_MA_COLORS = {20: "#1f6feb", 50: "#f59e0b", 100: "#14b8a6", 200: "#7c3aed"}
_MOMENTUM_COLORS = ["#1f6feb", "#f59e0b", "#14b8a6", "#7c3aed", "#db2777"]

# Distinct hues *and* dash patterns for the new-high / new-low lookbacks so the
# 20-D and 252-D lines are never mistaken for each other. Highs stay in the
# green family (bullish), lows in the red family (bearish); the dash pattern
# adds a second visual cue on top of colour.
_HIGH_COLORS = ["#2eab5a", "#65a30d", "#06b6d4"]
_LOW_COLORS = ["#d64545", "#9f1239", "#f97316"]
# Custom px dash strings ("dashlen,gaplen,...") give finer control than Plotly's
# built-in "dash"/"dashdot" presets. The 252-D lines (index 1) use a short
# dash + short gap so the dashes are markedly more frequent than the default
# "dash" preset; index 2 (the fallback 3-window case) is a denser dash-dot.
_LINE_DASHES = ["solid", "5px,3px", "5px,3px,1.5px,3px"]

_LABELS = {
    "breadth_score": "Breadth Score",
    "breadth_score_20d_ma": "20-Day MA",
    "breadth_score_percentile": "Percentile Rank",
    "pct_above_ma20": "% > 20-Day MA",
    "pct_above_ma50": "% > 50-Day MA",
    "pct_above_ma100": "% > 100-Day MA",
    "pct_above_ma200": "% > 200-Day MA",
    "pct_up_volume": "% Up Volume",
    "mcclellan_style": "McClellan-Style",
    "ad_line": "A/D Line",
    "index_close": "Index (SPY)",
    "advancers": "Advancers",
    "decliners": "Decliners",
    "net_advancers": "Net Advancers",
    "pct_new_high_20d": "New 20-D Highs",
    "pct_new_low_20d": "New 20-D Lows",
    "pct_new_high_63d": "New 63-D Highs",
    "pct_new_low_63d": "New 63-D Lows",
    "pct_new_high_252d": "New 252-D Highs",
    "pct_new_low_252d": "New 252-D Lows",
}


def _label(col: str) -> str:
    """Map a raw column name to a human-readable legend label."""
    return _LABELS.get(col, col.replace("pct_", "").replace("_", " ").title())


def _empty_figure(title: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, height=420, template="plotly_white")
    return fig

#---
def _date_axis_ticks(fig: go.Figure) -> Tuple[object, str, Optional[str]]:
    """Pick an x-axis tick step, label format and (optional) tick0 anchor.

    Plotly's auto-ticker spaces ticks by pixel width, not calendar boundaries,
    so a short range (e.g. 3 months) can place two ticks inside one month and
    render the same "%b %Y" label twice.  Anchoring ticks to a calendar step
    whose width exceeds the label granularity makes duplicates impossible.

    For short ranges we use a numeric millisecond step with a "%b %d" format
    and *also* return a ``tick0`` anchored to the first date in the figure.
    Without an explicit ``tick0`` Plotly anchors date-axis ticks at 2000-01-01
    (a Saturday); a 14-day step then lands every tick on a non-trading weekend
    that exists nowhere in the data. Anchoring at the first data date keeps
    every tick on a real weekday that matches the series. The anchor is
    formatted as a plain "YYYY-MM-DD" string, the canonical Plotly date tick0.

    Calendar month/year steps ("M1" ... "M12") already anchor to the 1st of a
    month, so no ``tick0`` is returned for the longer ranges.
    """
    xs = []
    for tr in fig.data:
        x = getattr(tr, "x", None)
        if x is None:
            continue
        try:
            xs.append(pd.to_datetime(pd.Series(x)))
        except Exception:
            continue
    if not xs:
        return None, "%b %Y", None
    all_x = pd.concat(xs).dropna()
    if all_x.empty:
        return None, "%b %Y", None
    span_days = int((all_x.max() - all_x.min()).days)

    if span_days <= 150:
        # Anchor ticks to the first date in the data so they fall on real
        # trading days instead of Plotly's 2000-01-01 (Saturday) default.
        return 14 * 24 * 3600 * 1000, "%b %d", all_x.min().strftime("%Y-%m-%d")
    if span_days <= 430:
        return "M1", "%b %Y", None
    if span_days <= 800:
        return "M2", "%b %Y", None
    if span_days <= 1500:
        return "M3", "%b %Y", None
    if span_days <= 3050:
        return "M6", "%b %Y", None
    return "M12", "%Y", None


def _apply_layout(
    fig: go.Figure,
    title: str,
    y_title: str = "",
    height: int = 420,
    x_title: str = "Date",
    y_range: Optional[Tuple[float, float]] = None,
    pct: bool = False,
    date_axis: bool = True,
) -> go.Figure:
    """Apply the shared visual style to a figure."""
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=70, b=50),
    )
    if date_axis:
        dtick, fmt, tick0 = _date_axis_ticks(fig)
        fig.update_xaxes(title_text=x_title, tickformat=fmt)
        x_axes_kwargs = {}
        if dtick is not None:
            x_axes_kwargs["dtick"] = dtick
        if tick0 is not None:
            x_axes_kwargs["tick0"] = tick0
        if x_axes_kwargs:
            fig.update_xaxes(**x_axes_kwargs)
    else:
        fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
    if pct:
        fig.update_yaxes(ticksuffix="%", tickformat=".0f")
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    return fig

#---

def _old_date_axis_ticks(fig: go.Figure) -> Tuple[object, str]:
    """Pick an x-axis tick step + label format so labels are never duplicated.

    Plotly's auto-ticker spaces ticks by pixel width, not calendar boundaries,
    so a short range (e.g. 3 months) can place two ticks inside one month and
    render the same "%b %Y" label twice.  Anchoring ticks to a calendar step
    whose width exceeds the label granularity makes duplicates impossible.
    """
    xs = []
    for tr in fig.data:
        x = getattr(tr, "x", None)
        if x is None:
            continue
        try:
            xs.append(pd.to_datetime(pd.Series(x)))
        except Exception:
            continue
    if not xs:
        return None, "%b %Y"
    all_x = pd.concat(xs).dropna()
    if all_x.empty:
        return None, "%b %Y"
    span_days = int((all_x.max() - all_x.min()).days)

    if span_days <= 150:
        return 14 * 24 * 3600 * 1000, "%b %d"
    if span_days <= 430:
        return "M1", "%b %Y"
    if span_days <= 800:
        return "M2", "%b %Y"
    if span_days <= 1500:
        return "M3", "%b %Y"
    if span_days <= 3050:
        return "M6", "%b %Y"
    return "M12", "%Y"


def _old_apply_layout(
    fig: go.Figure,
    title: str,
    y_title: str = "",
    height: int = 420,
    x_title: str = "Date",
    y_range: Optional[Tuple[float, float]] = None,
    pct: bool = False,
    date_axis: bool = True,
) -> go.Figure:
    """Apply the shared visual style to a figure."""
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=70, b=50),
    )
    if date_axis:
        dtick, fmt, tick0 = _date_axis_ticks(fig)
        fig.update_xaxes(title_text=x_title, tickformat=fmt)
        if dtick is not None:
            fig.update_xaxes(dtick=dtick)
        if tick0 is not None:
            fig.update_xaxes(tick0=tick0)
    else:
        fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
    if pct:
        fig.update_yaxes(ticksuffix="%", tickformat=".0f")
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    return fig


def _contiguous_runs(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (start, end) index pairs for contiguous True runs in a boolean series."""
    runs: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    idx, vals = mask.index, mask.to_numpy()
    in_run, start = False, None
    for i, v in enumerate(vals):
        if bool(v) and not in_run:
            start, in_run = idx[i], True
        elif not bool(v) and in_run:
            runs.append((start, idx[i - 1]))
            in_run = False
    if in_run:
        runs.append((start, idx[-1]))
    return runs


def _contiguous_runs_label(series: pd.Series):
    """Return (start, end, label) tuples for contiguous runs of equal values."""
    runs = []
    idx, vals = series.index, series.to_numpy()
    in_run, start, label = False, None, None
    for i, v in enumerate(vals):
        if pd.isna(v):
            if in_run:
                runs.append((start, idx[i - 1], label))
                in_run = False
            continue
        if not in_run:
            start, label, in_run = idx[i], v, True
        elif v != label:
            runs.append((start, idx[i - 1], label))
            start, label = idx[i], v
    if in_run:
        runs.append((start, idx[-1], label))
    return runs


def _rebase(series: pd.Series) -> pd.Series:
    """Rebase a series to percent change from its first non-NaN value."""
    s = series.dropna()
    if s.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    base = s.iloc[0]
    if base == 0 or pd.isna(base):
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series / base - 1.0) * 100.0


def _vrect_shape(x0, x1, color: str, opacity: float = 0.12) -> dict:
    """Build a full-height vertical rectangle shape dict (paper y-reference)."""
    return dict(
        type="rect",
        xref="x",
        yref="paper",
        x0=x0,
        x1=x1,
        y0=0,
        y1=1,
        fillcolor=color,
        opacity=opacity,
        line_width=0,
        layer="below",
    )


def _add_shapes_batch(fig: go.Figure, shapes: list) -> None:
    """Append a batch of shape dicts to the figure layout in a single update.

    Calling add_vrect() in a loop is O(n^2) because each call re-merges the
    full shapes list. Batching into one update_layout keeps it linear and
    keeps the dashboard responsive on multi-year data.
    """
    existing = list(fig.layout.shapes) if fig.layout.shapes else []
    fig.update_layout(shapes=existing + shapes)


def plot_breadth_score(breadth: pd.DataFrame, show_ma: bool = True) -> go.Figure:
    """Composite breadth score with threshold bands and a percentile overlay."""
    if "breadth_score" not in breadth.columns:
        return _empty_figure("Breadth score not available.")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=breadth.index,
            y=breadth["breadth_score"],
            name="Breadth Score",
            line=dict(color=COLORS["accent"], width=2),
            hovertemplate="%{y:.2f}<extra>Breadth Score</extra>",
        ),
        secondary_y=False,
    )

    if show_ma and "breadth_score_20d_ma" in breadth.columns:
        fig.add_trace(
            go.Scatter(
                x=breadth.index,
                y=breadth["breadth_score_20d_ma"],
                name="20-Day MA",
                line=dict(color=COLORS["neutral"], dash="dash", width=1.5),
                hovertemplate="%{y:.2f}<extra>20-Day MA</extra>",
            ),
            secondary_y=False,
        )

    if "breadth_score_percentile" in breadth.columns:
        fig.add_trace(
            go.Scatter(
                x=breadth.index,
                y=breadth["breadth_score_percentile"] * 100,
                name="Percentile Rank",
                line=dict(color=COLORS["bull"], dash="dot", width=1.5),
                opacity=0.7,
                hovertemplate="%{y:.0f}%<extra>Percentile Rank</extra>",
            ),
            secondary_y=True,
        )

    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["bull"], line_width=1)
    fig.add_hline(y=-0.5, line_dash="dash", line_color=COLORS["bear"], line_width=1)
    fig.add_hrect(y0=0.5, y1=4, fillcolor=COLORS["bull"], opacity=0.07, line_width=0)
    fig.add_hrect(y0=-4, y1=-0.5, fillcolor=COLORS["bear"], opacity=0.07, line_width=0)

    _apply_layout(fig, "Composite Breadth Score", "Z-Score", height=440)
    fig.update_yaxes(
        title_text="Percentile Rank",
        range=[0, 100],
        ticksuffix="%",
        tickformat=".0f",
        secondary_y=True,
    )
    return fig


def plot_percent_above_moving_averages(breadth: pd.DataFrame) -> go.Figure:
    """Percentage of stocks above key moving averages, with a 50% reference line."""
    cols = [c for c in breadth.columns if c.startswith("pct_above_ma")]
    if not cols:
        return _empty_figure("Moving-average breadth not available.")

    def _win(col: str) -> int:
        m = re.search(r"ma(\d+)", col)
        return int(m.group(1)) if m else 9999

    cols = sorted(cols, key=_win)
    fig = go.Figure()
    for col in cols:
        color = _MA_COLORS.get(_win(col), COLORS["accent"])
        fig.add_trace(
            go.Scatter(
                x=breadth.index,
                y=breadth[col] * 100,
                name=_label(col),
                line=dict(color=color, width=1.8),
                hovertemplate="%{y:.1f}%<extra>%{fullData.name}</extra>",
            )
        )

    fig.add_hline(y=50, line_dash="dash", line_color=COLORS["neutral"], line_width=1)
    _apply_layout(
        fig,
        "Percentage of Stocks Above Moving Averages",
        "Percent",
        y_range=(0, 100),
        pct=True,
    )
    return fig


def plot_advance_decline(breadth: pd.DataFrame) -> go.Figure:
    """Daily net advancers as green/red bars with a 10-day average line."""
    if "net_advancers" not in breadth.columns:
        return _empty_figure("Advance/decline data not available.")

    net = breadth["net_advancers"]
    bar_colors = [COLORS["bull"] if v >= 0 else COLORS["bear"] for v in net]
    ma = net.rolling(10, min_periods=5).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=net.index,
            y=net,
            name="Net Advancers",
            marker_color=bar_colors,
            hovertemplate="%{y:+d}<extra>Net Advancers</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ma.index,
            y=ma,
            name="10-Day MA",
            line=dict(color=COLORS["accent"], width=2),
            hovertemplate="%{y:.0f}<extra>10-Day MA</extra>",
        )
    )
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=1)
    _apply_layout(fig, "Daily Net Advancers (Advancers \u2212 Decliners)", "Number of Stocks")
    return fig


def plot_new_high_low(breadth: pd.DataFrame) -> go.Figure:
    """New highs (green, upward) vs new lows (red, mirrored downward).

    Each lookback window gets its own colour *and* dash pattern so the 20-D,
    63-D and 252-D lines are clearly distinguishable within a series. Highs use
    the green family, lows the red family, mirroring the bull/bear semantics.
    """
    preferred = [
        "pct_new_high_20d",
        "pct_new_low_20d",
        "pct_new_high_252d",
        "pct_new_low_252d",
    ]
    cols = [c for c in preferred if c in breadth.columns]
    if not cols:
        cols = [c for c in breadth.columns if c.startswith("pct_new")]
    if not cols:
        return _empty_figure("New high/low breadth not available.")

    def _win(col: str) -> int:
        m = re.search(r"(\d+)d", col)
        return int(m.group(1)) if m else 9999

    high_cols = sorted([c for c in cols if "high" in c], key=_win)
    low_cols = sorted([c for c in cols if "low" in c], key=_win)

    fig = go.Figure()
    for i, col in enumerate(high_cols):
        fig.add_trace(
            go.Scatter(
                x=breadth.index,
                y=breadth[col] * 100,
                name=_label(col),
                line=dict(
                    color=_HIGH_COLORS[i % len(_HIGH_COLORS)],
                    dash=_LINE_DASHES[i % len(_LINE_DASHES)],
                    width=1.8,
                ),
                hovertemplate="%{y:.2f}%<extra>%{fullData.name}</extra>",
            )
        )
    for i, col in enumerate(low_cols):
        fig.add_trace(
            go.Scatter(
                x=breadth.index,
                y=-breadth[col] * 100,
                name=_label(col),
                line=dict(
                    color=_LOW_COLORS[i % len(_LOW_COLORS)],
                    dash=_LINE_DASHES[i % len(_LINE_DASHES)],
                    width=1.8,
                ),
                hovertemplate="%{y:.2f}%<extra>%{fullData.name}</extra>",
            )
        )

    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=1)
    _apply_layout(fig, "New Highs vs New Lows (mirrored)", "Percent")
    return fig


def plot_volume_breadth(breadth: pd.DataFrame) -> go.Figure:
    """Up-volume as a share of total volume, with a 50% reference and 10-day MA."""
    if "pct_up_volume" not in breadth.columns:
        return _empty_figure("Volume breadth not available.")

    pv = breadth["pct_up_volume"] * 100
    ma = pv.rolling(10, min_periods=5).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pv.index,
            y=pv,
            name="% Up Volume",
            line=dict(color=COLORS["accent"], width=2),
            hovertemplate="%{y:.1f}%<extra>% Up Volume</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ma.index,
            y=ma,
            name="10-Day MA",
            line=dict(color=COLORS["index"], dash="dash", width=1.5),
            hovertemplate="%{y:.1f}%<extra>10-Day MA</extra>",
        )
    )
    fig.add_hline(y=50, line_dash="dash", line_color=COLORS["neutral"], line_width=1)
    _apply_layout(fig, "Up-Volume as % of Total Volume", "Percent", y_range=(0, 100), pct=True)
    return fig


def plot_index_vs_breadth(breadth: pd.DataFrame) -> go.Figure:
    """Index vs breadth score, rebased to % change, with divergence shading."""
    if "index_close" not in breadth.columns or "breadth_score" not in breadth.columns:
        return _empty_figure("Index vs breadth data not available.")

    idx_r = _rebase(breadth["index_close"])
    score_r = _rebase(breadth["breadth_score"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=breadth.index,
            y=idx_r,
            name="Index (SPY)",
            line=dict(color=COLORS["index"], width=2),
            hovertemplate="%{y:+.1f}%<extra>Index</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=breadth.index,
            y=score_r,
            name="Breadth Score",
            line=dict(color=COLORS["accent"], width=2),
            hovertemplate="%{y:+.1f}%<extra>Breadth</extra>",
        )
    )
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=1)

    vrects = []
    if "bearish_divergence" in breadth.columns:
        for s, e in _contiguous_runs(breadth["bearish_divergence"].astype(bool)):
            vrects.append(_vrect_shape(s, e, COLORS["bear"]))
    if "bullish_divergence" in breadth.columns:
        for s, e in _contiguous_runs(breadth["bullish_divergence"].astype(bool)):
            vrects.append(_vrect_shape(s, e, COLORS["bull"]))
    _add_shapes_batch(fig, vrects)

    _apply_layout(
        fig,
        "Index vs Breadth Score (rebased to % change)",
        "Change since window start (%)",
    )
    return fig


def plot_ad_line_vs_index(breadth: pd.DataFrame) -> go.Figure:
    """Cumulative A/D line vs index, both rebased to % change."""
    if "ad_line" not in breadth.columns or "index_close" not in breadth.columns:
        return _empty_figure("A/D line or index data not available.")

    ad_r = _rebase(breadth["ad_line"])
    idx_r = _rebase(breadth["index_close"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=breadth.index,
            y=idx_r,
            name="Index (SPY)",
            line=dict(color=COLORS["index"], width=2),
            hovertemplate="%{y:+.1f}%<extra>Index</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=breadth.index,
            y=ad_r,
            name="A/D Line",
            line=dict(color=COLORS["accent"], width=2),
            hovertemplate="%{y:+.1f}%<extra>A/D Line</extra>",
        )
    )
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=1)
    _apply_layout(fig, "Cumulative A/D Line vs Index (rebased)", "Change since window start (%)")
    return fig


def plot_mcclellan(breadth: pd.DataFrame) -> go.Figure:
    """McClellan-style oscillator as a green/red histogram around zero."""
    if "mcclellan_style" not in breadth.columns:
        return _empty_figure("McClellan-style oscillator not available.")

    mc = breadth["mcclellan_style"]
    bar_colors = [COLORS["bull"] if v >= 0 else COLORS["bear"] for v in mc]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=mc.index,
            y=mc,
            name="McClellan-Style",
            marker_color=bar_colors,
            hovertemplate="%{y:.3f}<extra>McClellan</extra>",
        )
    )
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=1)
    _apply_layout(fig, "McClellan-Style Oscillator", "Value")
    return fig


def plot_regime_timeline(breadth: pd.DataFrame) -> go.Figure:
    """Breadth score with background bands coloured by the daily regime."""
    if "breadth_score" not in breadth.columns:
        return _empty_figure("Breadth score not available.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=breadth.index,
            y=breadth["breadth_score"],
            name="Breadth Score",
            line=dict(color=COLORS["accent"], width=2),
            hovertemplate="%{y:.2f}<extra>Breadth Score</extra>",
        )
    )

    if "breadth_regime" in breadth.columns:
        regime_colors = {
            "Broadening": COLORS["bull"],
            "Narrowing": COLORS["bear"],
            "Neutral": COLORS["neutral"],
        }
        vrects = [
            _vrect_shape(s, e, regime_colors.get(label, COLORS["neutral"]))
            for s, e, label in _contiguous_runs_label(breadth["breadth_regime"])
        ]
        _add_shapes_batch(fig, vrects)

    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["bull"], line_width=1)
    fig.add_hline(y=-0.5, line_dash="dash", line_color=COLORS["bear"], line_width=1)
    _apply_layout(fig, "Breadth Regime Timeline", "Z-Score")
    return fig


def plot_momentum(breadth: pd.DataFrame) -> go.Figure:
    """Percentage of stocks with positive returns over each lookback window."""
    cols = [c for c in breadth.columns if c.startswith("pct_positive_")]
    if not cols:
        return _empty_figure("Momentum breadth not available.")

    def _win(col: str) -> int:
        m = re.search(r"(\d+)d", col)
        return int(m.group(1)) if m else 9999

    cols = sorted(cols, key=_win)
    fig = go.Figure()
    for i, col in enumerate(cols):
        fig.add_trace(
            go.Scatter(
                x=breadth.index,
                y=breadth[col] * 100,
                name=f"{_win(col)}-Day",
                line=dict(color=_MOMENTUM_COLORS[i % len(_MOMENTUM_COLORS)], width=1.8),
                hovertemplate="%{y:.1f}%<extra>%{fullData.name} positive</extra>",
            )
        )
    fig.add_hline(y=50, line_dash="dash", line_color=COLORS["neutral"], line_width=1)
    _apply_layout(
        fig,
        "% of Stocks with Positive Returns Over Lookback",
        "Percent",
        y_range=(0, 100),
        pct=True,
    )
    return fig
# ---------------------------------------------------------------------------
# Sector rotation charts
# ---------------------------------------------------------------------------

SECTOR_COLORS = [
    "#1f6feb", "#eab308", "#16a34a", "#dc2626", "#7c3aed",
    "#14b8a6", "#db2777", "#84cc16", "#f97316", "#92400e", "#475569",
]


def _sector_color(i: int) -> str:
    return SECTOR_COLORS[i % len(SECTOR_COLORS)]


def plot_sector_rs_lines(
    sector_index: pd.DataFrame,
    index_ticker: str = "SPY",
) -> go.Figure:
    """Relative strength of each sector vs the benchmark, rebased to 0%."""
    if sector_index is None or sector_index.empty:
        return _empty_figure("Sector index data not available.")
    if index_ticker not in sector_index.columns:
        return _empty_figure("Benchmark column not found in sector index.")

    bench = sector_index[index_ticker]
    sector_cols = [
        c for c in sector_index.columns if c != index_ticker and c != "def_cyc_ratio"
    ]
    if not sector_cols:
        return _empty_figure("No sector columns available.")

    fig = go.Figure()
    for i, sector in enumerate(sorted(sector_cols)):
        rs = sector_index[sector] / bench
        fig.add_trace(
            go.Scatter(
                x=sector_index.index,
                y=_rebase(rs),
                name=sector,
                line=dict(color=_sector_color(i), width=1.8),
                hovertemplate="%{y:+.1f}%<extra>" + sector + "</extra>",
            )
        )
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=1)
    _apply_layout(
        fig,
        "Sector Relative Strength vs " + index_ticker + " (rebased)",
        "Outperformance since window start (%)",
    )
    return fig


def plot_defensive_vs_cyclical(sector_index: pd.DataFrame) -> go.Figure:
    """Defensive / Cyclical ratio as a risk-on / risk-off gauge."""
    if sector_index is None or "def_cyc_ratio" not in (sector_index.columns if sector_index is not None else []):
        return _empty_figure("Defensive/Cyclical ratio not available.")

    ratio = sector_index["def_cyc_ratio"]
    ma = ratio.rolling(200, min_periods=63).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ratio.index,
            y=ratio,
            name="Defensive / Cyclical",
            line=dict(color=COLORS["accent"], width=2),
            hovertemplate="%{y:.3f}<extra>Def / Cyc</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ma.index,
            y=ma,
            name="200-Day MA",
            line=dict(color=COLORS["neutral"], dash="dash", width=1.5),
            hovertemplate="%{y:.3f}<extra>200-Day MA</extra>",
        )
    )
    fig.add_hline(y=1.0, line_color=COLORS["bear"], line_dash="dot", line_width=1)
    _apply_layout(fig, "Defensive vs Cyclical Ratio (risk-on / risk-off)", "Ratio (Defensive / Cyclical)")
    return fig
def plot_sector_breadth_heatmap(
    breadth_long: pd.DataFrame,
    metric: str = "pct_above_ma50",
    max_cols: int = 400,
) -> go.Figure:
    """Sector x date heatmap of one breadth metric (downsampled for speed)."""
    if breadth_long is None or breadth_long.empty or metric not in breadth_long.columns:
        return _empty_figure(f"Sector breadth for {metric} not available.")

    piv = (breadth_long[metric].unstack("sector") * 100.0).sort_index()
    if len(piv) > max_cols:
        piv = piv.iloc[:: int(np.ceil(len(piv) / max_cols))]
    sectors = sorted(piv.columns)
    piv = piv[sectors]

    fig = go.Figure(
        data=go.Heatmap(
            z=piv.T.values,
            x=piv.index,
            y=sectors,
            colorscale="RdYlGn",
            zmid=50,
            zmin=0,
            zmax=100,
            hovertemplate="%{y}<br>%{x|%b %Y}: %{z:.0f}%<extra></extra>",
            colorbar=dict(title="%", ticksuffix="%"),
        )
    )
    _apply_layout(fig, f"Sector Breadth Heatmap \u2014 {_label(metric)}", height=460)
    return fig


def plot_sector_momentum_matrix(breadth_long: pd.DataFrame) -> go.Figure:
    """Latest per-sector breadth across metrics, as a coloured matrix."""
    if breadth_long is None or breadth_long.empty:
        return _empty_figure("Sector breadth not available.")

    metrics = [
        "pct_above_ma50",
        "pct_above_ma200",
        "pct_positive_20d",
        "pct_positive_63d",
        "pct_advancers",
        "pct_up_volume",
    ]
    avail = [m for m in metrics if m in breadth_long.columns]
    if not avail:
        return _empty_figure("No sector breadth metrics available.")

    latest = breadth_long.groupby(level="sector").tail(1).reset_index(level="sector")
    matrix = (latest.set_index("sector")[avail] * 100.0).sort_index()
    z = matrix.values
    text = [[f"{v:.0f}" if pd.notna(v) else "" for v in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[_label(m) for m in avail],
            y=list(matrix.index),
            colorscale="RdYlGn",
            zmid=50,
            zmin=0,
            zmax=100,
            text=text,
            texttemplate="%{text}%",
            hovertemplate="%{y} | %{x}: %{z:.0f}%<extra></extra>",
            colorbar=dict(title="%", ticksuffix="%"),
        )
    )
    _apply_layout(fig, "Sector Breadth Matrix (latest values)", height=420, date_axis=False)
    return fig