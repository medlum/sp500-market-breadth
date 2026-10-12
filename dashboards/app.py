from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.breadth import config
from src.breadth.storage import load_breadth
from src.breadth.plots import (
    plot_ad_line_vs_index,
    plot_advance_decline,
    plot_breadth_score,
    plot_defensive_vs_cyclical,
    plot_index_vs_breadth,
    plot_mcclellan,
    plot_momentum,
    plot_new_high_low,
    plot_percent_above_moving_averages,
    plot_regime_timeline,
    plot_sector_breadth_heatmap,
    plot_sector_momentum_matrix,
    plot_sector_rs_lines,
    plot_volume_breadth,
)
from src.breadth.sectors import rotation_summary, rs_rank, sector_relative_strength

st.set_page_config(
    page_title="S&P 500 Market Breadth",
    page_icon="\U0001F4CA",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Plain-English captions shown beneath each chart
# ---------------------------------------------------------------------------

CAPTIONS = {
    "regime": (
        "Background shaded by the daily regime: green = Broadening, red = Narrowing, "
        "gray = Neutral. Regimes flip as the score crosses \u00B10.5."
    ),
    "score": (
        "Z-score blending ~13 breadth measures vs the prior year. Above +0.5 (green band) "
        "\u21D2 participation is broad; below \u22120.5 (red band) \u21D2 narrowing. "
        "The dotted line is the percentile rank (0\u2013100%)."
    ),
    "index_vs": (
        "Both series rebased to 0% at the start of the selected window. Red shading = bearish "
        "divergence (index rising while breadth falls); green = bullish divergence."
    ),
    "above_ma": (
        "Share of S&P 500 stocks above each moving average. Above 50% \u21D2 most stocks are "
        "in an uptrend over that lookback."
    ),
    "momentum": (
        "Share of stocks with a positive return over each lookback. Above 50% \u21D2 most "
        "stocks are net positive over that period."
    ),
    "high_low": (
        "New highs plotted upward (green), new lows mirrored downward (red). A healthy market "
        "shows highs dominating and lows near zero."
    ),
    "ad": (
        "Green bars \u21D2 more advancers than decliners that day; red \u21D2 the opposite. "
        "The blue line is the 10-day average of net advancers."
    ),
    "ad_line": (
        "Cumulative net advancers vs the index, both rebased to 0%. The A/D line keeping pace "
        "with or leading the index up is healthy confirmation."
    ),
    "mcclellan": (
        "Difference of two EMAs of net advancers. Positive (green) \u21D2 short-term breadth "
        "momentum is expanding; negative (red) \u21D2 contracting."
    ),
    "volume": (
        "Share of total volume in advancing stocks. Above 50% \u21D2 buyers are in control that day."
    ),
    "sector_rs": (
        "Each sector's relative strength vs SPY, rebased to 0% at the start of the window. "
        "Rising \u21D2 sector outperforming the market; falling \u21D2 lagging."
    ),
    "sector_heatmap": (
        "Share of each sector's stocks above their 50-day MA, over time. Green \u21D2 broad "
        "participation; red \u21D2 most stocks below the MA."
    ),
    "sector_defcyc": (
        "Defensive sectors (Staples + Utilities + Health Care) divided by cyclical sectors "
        "(Discretionary + Industrials + Materials). Rising \u21D2 risk-off rotation; "
        "falling \u21D2 risk-on."
    ),
    "sector_matrix": (
        "Latest breadth per sector across lookbacks. Greens \u21D2 healthy; reds \u21D2 weak. "
        "Spot which sectors are leading/lagging right now."
    ),
    "sector_rotation": (
        "Each sector ranked by current relative strength (1 = strongest). A rank jump of "
        "\u22653 places over the last quarter flags a rotation in (green) or out (red)."
    ),
}


def _fmt_pct(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.1%}"


def _fmt_number(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"


def _safe_delta(curr, prev):
    if pd.isna(curr) or pd.isna(prev):
        return None
    return float(curr) - float(prev)


def _as_bool(value) -> bool:
    return pd.notna(value) and bool(value)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

if not config.BREADTH_OUTPUT.exists():
    st.warning("Breadth output not found. Run the pipeline first.")
    st.code("python run_pipeline.py", language="bash")
    st.stop()


@st.cache_data(ttl=3600)
def get_breadth() -> pd.DataFrame:
    df = load_breadth(config.BREADTH_OUTPUT)
    return df.sort_index()


breadth_full = get_breadth()

if breadth_full.empty:
    st.warning("Breadth table is empty.")
    st.stop()


@st.cache_data(ttl=3600)
def get_sector_data():
    """Load sector index + long breadth, and derive RS/ranks/rotation summary."""
    if not config.SECTOR_INDEX_OUTPUT.exists() or not config.SECTOR_BREADTH_OUTPUT.exists():
        return None
    sector_index = pd.read_parquet(config.SECTOR_INDEX_OUTPUT).sort_index()
    breadth_long = pd.read_parquet(config.SECTOR_BREADTH_OUTPUT).sort_index()
    if sector_index.empty or breadth_long.empty:
        return None
    # Defensive: normalise level names from older saved files (the first level
    # may be 'Date' instead of 'date').
    if isinstance(breadth_long.index, pd.MultiIndex) and breadth_long.index.nlevels == 2:
        breadth_long.index = breadth_long.index.set_names(["date", "sector"])
    bench = sector_index[config.INDEX_TICKER] if config.INDEX_TICKER in sector_index.columns else None
    rs = (
        sector_relative_strength(
            sector_index.drop(columns=[c for c in [config.INDEX_TICKER, "def_cyc_ratio"] if c in sector_index.columns]),
            bench,
        )
        if bench is not None
        else pd.DataFrame()
    )
    ranks = rs_rank(rs) if not rs.empty else pd.DataFrame()
    return {
        "sector_index": sector_index,
        "breadth_long": breadth_long,
        "rs": rs,
        "ranks": ranks,
        "rotation": rotation_summary(rs, ranks) if not ranks.empty else pd.DataFrame(),
    }


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("\u2699\uFE0F Controls")
    range_options = {
        "6 Months": 180,
        "1 Year": 365,
        "2 Years": 730,
        "5 Years": 1825,
        "All history": None,
    }
    sel = st.selectbox("Date range", list(range_options.keys()), index=1)
    days = range_options[sel]
    smoothing = st.checkbox("Show 20-Day MA on breadth score", value=True)
    st.divider()
    last_valid = (
        int(breadth_full["valid_stocks"].iloc[-1])
        if "valid_stocks" in breadth_full.columns
        else 0
    )
    st.caption(f"**Universe:** ~{last_valid} stocks")
    st.caption(
        f"**History:** {breadth_full.index.min():%Y-%m-%d} \u2192 {breadth_full.index.max():%Y-%m-%d}"
    )
    st.caption("**Benchmark:** SPY")

# ---------------------------------------------------------------------------
# Apply the selected date range
# ---------------------------------------------------------------------------

if days is None:
    breadth = breadth_full
else:
    cutoff = breadth_full.index.max() - pd.Timedelta(days=days)
    breadth = breadth_full.loc[breadth_full.index >= cutoff]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("\U0001F4CA S&P 500 Market Breadth Monitor")
st.caption(f"As of {breadth.index.max():%A, %B %d, %Y}")

latest = breadth.dropna(subset=["breadth_score"]).tail(1)
if latest.empty:
    st.warning("No breadth score is available in the selected range.")
    st.stop()

row = latest.iloc[0]

scored = breadth_full.dropna(subset=["breadth_score"]).tail(2)
prev_row = scored.iloc[0] if len(scored) >= 2 else row

regime_val = row.get("breadth_regime", "N/A")
regime = "N/A" if pd.isna(regime_val) else str(regime_val)

regime_msgs = {
    "Broadening": ("\U0001F7E2 **Broadening** \u2014 participation is expanding across the index.", "success"),
    "Narrowing": ("\U0001F534 **Narrowing** \u2014 breadth is deteriorating; the rally may be thin.", "error"),
    "Neutral": ("\U0001F7E1 **Neutral** \u2014 breadth is mixed; no strong directional signal.", "warning"),
}
msg, kind = regime_msgs.get(regime, (f"**Regime:** {regime}", "info"))
getattr(st, kind)(msg)

# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------

score_delta = _safe_delta(row.get("breadth_score"), prev_row.get("breadth_score"))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "Breadth Score",
    _fmt_number(row.get("breadth_score")),
    delta=round(score_delta, 2) if score_delta is not None else None,
)
c2.metric("Percentile Rank", _fmt_pct(row.get("breadth_score_percentile")))
c3.metric("Regime", regime)
c4.metric("% Advancers", _fmt_pct(row.get("pct_advancers")))
c5.metric("% Above 50-Day MA", _fmt_pct(row.get("pct_above_ma50")))

st.divider()

# ---------------------------------------------------------------------------
# Divergence banners
# ---------------------------------------------------------------------------

if _as_bool(row.get("bearish_divergence")):
    st.error("\u26A0\uFE0F **Bearish divergence today:** SPY is up over 20 days, but the breadth score is falling.")

if _as_bool(row.get("bullish_divergence")):
    st.success("\u2705 **Bullish divergence today:** SPY is down over 20 days, but the breadth score is rising.")


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------

with st.expander("\U0001F4D6 How to read this dashboard"):
    st.markdown(
        """
**Breadth Score** \u2014 a rolling z-score that averages ~13 breadth indicators against
the prior year. It summarises whether participation is broad (positive) or narrow
(negative). Bands at \u00B10.5 mark the Broadening / Narrowing thresholds.

**Percentile Rank** \u2014 where today's score sits within the last year (0\u2013100%).
50% = average; near 0% or 100% = extreme.

**% Above MA** \u2014 share of stocks trading above their N-day moving average. A reading
above 50% means most stocks are in an uptrend over that lookback.

**Net Advancers** \u2014 advancers minus decliners per day. The cumulative version is the
**A/D Line**, a classic breadth confirmation tool.

**McClellan-Style Oscillator** \u2014 the difference of two EMAs of net advancers. Positive \u21D2
short-term breadth momentum is expanding.

**New Highs / Lows** \u2014 share of stocks making rolling N-day highs/lows. Lows near zero
with rising highs is healthy.

**Divergence** \u2014 when the index and breadth disagree. Bearish: index up but breadth
down (rally lacks participation). Bullish: index down but breadth up (selling is broad
but internals are healing).
"""
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_trend, tab_momentum, tab_hl, tab_adv, tab_sectors, tab_data = st.tabs(
    ["Overview", "Trend", "Momentum", "Highs / Lows", "A/D & Volume", "Sectors", "Data"]
)

with tab_overview:
    st.plotly_chart(plot_regime_timeline(breadth), width="stretch")
    st.caption(CAPTIONS["regime"])
    st.plotly_chart(plot_breadth_score(breadth, show_ma=smoothing), width="stretch")
    st.caption(CAPTIONS["score"])
    st.plotly_chart(plot_index_vs_breadth(breadth), width="stretch")
    st.caption(CAPTIONS["index_vs"])

with tab_trend:
    st.plotly_chart(plot_percent_above_moving_averages(breadth), width="stretch")
    st.caption(CAPTIONS["above_ma"])

with tab_momentum:
    st.plotly_chart(plot_momentum(breadth), width="stretch")
    st.caption(CAPTIONS["momentum"])

with tab_hl:
    st.plotly_chart(plot_new_high_low(breadth), width="stretch")
    st.caption(CAPTIONS["high_low"])

with tab_adv:
    st.plotly_chart(plot_advance_decline(breadth), width="stretch")
    st.caption(CAPTIONS["ad"])
    st.plotly_chart(plot_ad_line_vs_index(breadth), width="stretch")
    st.caption(CAPTIONS["ad_line"])
    st.plotly_chart(plot_mcclellan(breadth), width="stretch")
    st.caption(CAPTIONS["mcclellan"])
    st.plotly_chart(plot_volume_breadth(breadth), width="stretch")
    st.caption(CAPTIONS["volume"])

with tab_sectors:
    sec = get_sector_data()
    if sec is None:
        st.info(
            "Sector data hasn't been built yet. Re-run the pipeline to enable "
            "sector-rotation signals:"
        )
        st.code("python run_pipeline.py", language="bash")
    else:
        start = breadth.index.min()
        si = sec["sector_index"].loc[sec["sector_index"].index >= start]
        bl = sec["breadth_long"]
        bl = bl[bl.index.get_level_values("date") >= start]

        st.plotly_chart(plot_sector_rs_lines(si, config.INDEX_TICKER), width="stretch")
        st.caption(CAPTIONS["sector_rs"])

        st.plotly_chart(plot_defensive_vs_cyclical(si), width="stretch")
        st.caption(CAPTIONS["sector_defcyc"])

        st.plotly_chart(plot_sector_breadth_heatmap(bl), width="stretch")
        st.caption(CAPTIONS["sector_heatmap"])

        st.plotly_chart(plot_sector_momentum_matrix(bl), width="stretch")
        st.caption(CAPTIONS["sector_matrix"])

        st.subheader("Recent Rotation Signals")
        st.caption(CAPTIONS["sector_rotation"])
        rot = sec["rotation"].copy()
        if not rot.empty:
            st.dataframe(rot, width="stretch", hide_index=True)
        else:
            st.info("Not enough history to compute rotation signals yet.")

with tab_data:
    st.download_button(
        "\u2B07\uFE0F Download selected range as CSV",
        breadth.sort_index(ascending=False).to_csv().encode("utf-8"),
        file_name=f"sp500_breadth_{breadth.index.max():%Y%m%d}.csv",
        mime="text/csv",
    )
    st.dataframe(breadth.sort_index(ascending=False).head(100), width="stretch")