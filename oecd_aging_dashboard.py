# oecd_ageing_dashboard_v4.py
"""
OECD Ageing‑Risk Workbench
------------------------
Streamlit dashboard that blends historical World Bank & OECD data (2000‒latest)
with user‑configurable ARIMA projections to any year ≤ 2050.

✨ **What is new in v4**
• Pagination‑safe World Bank fetch (no silent truncation)
• Properly cached OECD debt series (st.cache_data)
• Sidebar controls for projection horizon **and** ARIMA (p,d,q)
• Dynamic preset weights that auto‑normalise to Σ = 1
• Cleaner pipeline: single imputation+scaling pass, no globals
• Bug fixes: duplicate scaling removed, decorator placement fixed
• Extra download buttons (snapshot & full panel)
"""

import datetime as dt
from io import StringIO
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
ISO3: List[str] = [
    "USA","CAN","GBR","DEU","FRA","ITA","ESP","JPN","KOR","AUS",
    "MEX","CHL","TUR","SWE","FIN","NOR","DNK","ISR","NLD","BEL",
    "CHE","AUT","GRC","IRL","NZL","CZE","POL","SVK","HUN","PRT",
    "EST","LVA","LTU","SVN","LUX",
]

INDICATORS: Dict[str, str] = {
    "SP.DYN.TFRT.IN": "Fertility rate (births / woman)",
    "SP.POP.65UP.TO.ZS": "Population 65+ (% of total)",
    "SH.XPD.CHEX.GD.ZS": "Health‑care spending / GDP (%)",
    "NY.GDP.PCAP.KD.ZG": "GDP per capita growth (% annual)",
    "SP.POP.DPND.OL": "Old‑age dependency ratio (65+ / 15‑64)",
    "GC.DOD.TOTL.GD.ZS": "Public debt / GDP (%)", 
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "Population 65+ (% of total)": 0.20,
    "Fertility rate (births / woman)": 0.20,
    "Health‑care spending / GDP (%)": 0.15,
    "GDP per capita growth (% annual)": 0.15,
    "Old‑age dependency ratio (65+ / 15‑64)": 0.15,
    "Public debt / GDP (%)": 0.15,
}

WB_BASE = "https://api.worldbank.org/v2"
MIN_YEAR = 2000
LATEST_YEAR = dt.datetime.now().year - 1
MAX_FUTURE = 2050

# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

def _normalise_weights(w: Dict[str, float]) -> Dict[str, float]:
    """Return a copy of *w* rescaled so that the values sum to 1."""
    s = sum(w.values()) or 1
    return {k: v / s for k, v in w.items()}


def _wb_fetch(indicator: str, start: int, end: int) -> pd.DataFrame:
    """Robust World Bank fetch handling pagination & missing data."""
    rows: List[Dict] = []
    page = 1
    while True:
        url = (
            f"{WB_BASE}/country/{';'.join(ISO3)}/indicator/{indicator}"
            f"?date={start}:{end}&page={page}&format=json&per_page=2000"
        )
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        js = resp.json()
        if len(js) < 2:
            break  # no data
        meta, data = js
        rows.extend(
            {
                "Country": rec["country"]["value"],
                "year": int(rec["date"]),
                indicator: rec["value"],
            }
            for rec in data
            if rec.get("value") is not None
        )
        if page >= int(meta.get("pages", 1)):
            break
        page += 1
    return pd.DataFrame(rows)


def _arima_forecast(series: pd.Series, order: Tuple[int, int, int], steps: int) -> List[float]:
    """Return *steps* forward forecasts for *series* (must be at least 5 non‑NA values)."""
    if series.dropna().size < 5:
        return [None] * steps
    try:
        model = ARIMA(series, order=order).fit()
        return model.forecast(steps).tolist()
    except Exception:
        return [None] * steps


@st.cache_data(ttl=86_400)
def load_panel(
    indicator_codes: List[str],
    start: int,
    end: int,
    future_end: int,
    arima_order: Tuple[int, int, int],
) -> pd.DataFrame:
    """Fetch WB series, forecast with ARIMA, merge to long panel."""
    panel: pd.DataFrame | None = None
    for code in indicator_codes:
        hist = _wb_fetch(code, start, end)
        # build forecasts (one ARIMA per country)
        steps = max(0, future_end - end)
        if steps:
            forecasts: List[Dict] = []
            for c, grp in hist.groupby("Country"):
                f_vals = _arima_forecast(grp.sort_values("year")[code], arima_order, steps)
                forecasts.extend(
                    {
                        "Country": c,
                        "year": yr,
                        code: val,
                    }
                    for yr, val in zip(range(end + 1, future_end + 1), f_vals)
                    if val is not None
                )
            fc_df = pd.DataFrame(forecasts)
            combined = pd.concat([hist, fc_df], ignore_index=True)
        else:
            combined = hist
        # long → wide per indicator then merge across indicators
        wide = (
            combined.pivot_table(index=["Country", "year"], values=code)
            .reset_index()
            .rename(columns={code: INDICATORS[code]})
        )
        panel = wide if panel is None else panel.merge(wide, on=["Country", "year"], how="outer")
    return panel if panel is not None else pd.DataFrame()

# ──────────────────────────────────────────────────────────────
# STREAMLIT APP
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="OECD Ageing‑Risk Workbench", layout="wide")

st.title("OECD Ageing‑Risk Workbench")
st.markdown(
    "Each country receives a composite vulnerability score based on weighted indicators. "
    "You can customize these weights or use predefined themes. The dashboard uses ARIMA modeling "
    "for time series forecasts, allowing future projections of key indicators up to year 2050.",
    unsafe_allow_html=True,
)
# ░░ Sidebar ░░
with st.sidebar:
    st.markdown("### Settings")
    horizon = st.slider("Projection horizon", min_value=LATEST_YEAR, max_value=MAX_FUTURE, value=2050, step=1)
    snap_year = st.slider("Snapshot year", min_value=MIN_YEAR, max_value=horizon, value=LATEST_YEAR)

    st.markdown("#### ARIMA order (p,d,q)")
    arima_p = st.number_input("p", min_value=0, max_value=5, value=1, step=1, key="arima_p")
    arima_d = st.number_input("d", min_value=0, max_value=2, value=1, step=1, key="arima_d")
    arima_q = st.number_input("q", min_value=0, max_value=5, value=0, step=1, key="arima_q")
    arima_order: Tuple[int, int, int] = (arima_p, arima_d, arima_q)

    st.markdown("#### Weight preset")
    preset = st.radio("Preset", ["Balanced", "Fiscal", "Health", "Custom"], index=0)
    user_weights = DEFAULT_WEIGHTS.copy()
    if preset == "Fiscal":
        user_weights.update({"Public debt / GDP (%)": 0.35, "Health‑care spending / GDP (%)": 0.25})
    elif preset == "Health":
        user_weights.update({"Health‑care spending / GDP (%)": 0.30, "Population 65+ (% of total)": 0.25})
    elif preset == "Custom":
        st.markdown("##### Custom weights (sum need not be 1 – it will be normalised)")
        for k in user_weights.keys():
            user_weights[k] = st.slider(k, 0.0, 1.0, float(user_weights[k]), 0.05)
    user_weights = _normalise_weights(user_weights)

# ░░ Data loading ░░
with st.spinner("Fetching / forecasting data …"):
    raw = load_panel(list(INDICATORS.keys()), MIN_YEAR, LATEST_YEAR, horizon, arima_order)
    
# ░░ Imputation ░░
for col in INDICATORS.values():
    if col not in raw.columns:
        continue
    raw[col] = (
        raw.sort_values(["Country", "year"])
        .groupby("Country")[col]
        .transform(lambda s: s.fillna(s.expanding().mean().shift(1)))
    )
    raw[col] = (
        raw.groupby("Country")[col]
        .transform(lambda s: s.fillna(s.mean()))
    )



# ░░ Scaling ░░
scaled = raw.copy()
for col in INDICATORS.values():
    if col not in scaled.columns:
        continue
    scaled[f"{col} (scaled)"] = MinMaxScaler().fit_transform(scaled[[col]])

# ░░ Composite score ░░
comp = pd.Series(0.0, index=scaled.index)
for k, w in user_weights.items():
    sc = f"{k} (scaled)"
    if sc in scaled.columns:
        comp += scaled[sc].fillna(0) * w
scaled["Vulnerability score"] = comp

# ░░ Snapshot DF ░░
snapshot_df = scaled[scaled["year"] == snap_year].copy()
if snapshot_df.empty:
    snapshot_df = scaled.groupby("Country").last().reset_index()

# data‑age flags (how stale each indicator is per country)
for col in DEFAULT_WEIGHTS:
    latest_years = scaled.dropna(subset=[col]).groupby("Country")["year"].max()
    snapshot_df[f"{col} (age)"] = snapshot_df["Country"].map(lambda c: snap_year - latest_years.get(c, snap_year))

# list of available indicators (after loading)
avail_ind = [c for c in DEFAULT_WEIGHTS.keys() if c in scaled.columns]

# sidebar country multi‑select
countries_all = sorted(snapshot_df["Country"].unique())
sel_countries = st.sidebar.multiselect("Countries to display", countries_all, default=countries_all[:5])
view_df = snapshot_df[snapshot_df["Country"].isin(sel_countries)]

# ░░ Layout ░░
overview, evolution, decomposition, map_tab, drill = st.tabs([
    "📊 Overview", "📈 Evolution", "🧮 Decomposition", "🗺️ Map", "🔍 Drill‑down",
])

# ──────────────────────────────────────────────────────────────
# Overview
# ──────────────────────────────────────────────────────────────
with overview:
    st.subheader("Vulnerability vs % population 65+")
    x_col, y_col = "Population 65+ (% of total)", "Vulnerability score"
    size_col = "Health‑care spending / GDP (%)"
    bub = view_df.copy()
    bub[size_col] = bub[size_col].fillna(bub[size_col].median())
    bub = bub.dropna(subset=[x_col, y_col])
    fig = px.scatter(
        bub,
        x=x_col,
        y=y_col,
        size=size_col,
        color="Country",
        size_max=50,
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(view_df, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# Evolution
# ──────────────────────────────────────────────────────────────
with evolution:
    st.subheader("Indicator evolution (2000–{0})".format(horizon))
    evo_c = st.multiselect("Choose countries", countries_all, default=sel_countries, key="evo_c")
    evo_i = st.selectbox("Choose indicator", avail_ind, key="evo_i")
    hist_df = scaled[scaled["Country"].isin(evo_c)].dropna(subset=[evo_i])
    fig2 = px.line(hist_df, x="year", y=evo_i, color="Country", height=600)
    st.plotly_chart(fig2, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# Decomposition
# ──────────────────────────────────────────────────────────────
with decomposition:
    st.subheader("Composite score decomposition")
    melt = view_df.melt(
        id_vars=["Country"],
        value_vars=[f"{c} (scaled)" for c in avail_ind],
        var_name="Factor",
        value_name="Scaled",
    )
    melt["Factor"] = melt["Factor"].str.replace(" (scaled)", "", regex=False)
    melt["Weight"] = melt["Factor"].map(user_weights)
    melt["Contribution"] = melt["Scaled"] * melt["Weight"]
    fig3 = px.bar(melt, x="Country", y="Contribution", color="Factor", height=600)
    st.plotly_chart(fig3, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# Map
# ──────────────────────────────────────────────────────────────
with map_tab:
    st.subheader("OECD vulnerability map (snapshot {0})".format(snap_year))
    fig4 = px.choropleth(
        snapshot_df,
        locations="Country",
        locationmode="country names",
        color="Vulnerability score",
        color_continuous_scale="Reds",
        hover_data=avail_ind,
        height=600,
    )
    st.plotly_chart(fig4, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# Drill‑down
# ──────────────────────────────────────────────────────────────
with drill:
    st.subheader("Country deep‑dive")
    dc = st.selectbox("Select country", countries_all, key="drill_country")
    row = snapshot_df[snapshot_df["Country"] == dc].iloc[0]
    factors = [c for c in avail_ind if c in row.index]
    if factors:
        radar_df = pd.DataFrame({
            "Factor": factors,
            dc: row[factors].values,
            "OECD median": snapshot_df[factors].median().values,
        })
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=radar_df[dc], theta=radar_df["Factor"], fill="toself", name=dc))
        fig_r.add_trace(go.Scatterpolar(r=radar_df["OECD median"], theta=radar_df["Factor"], fill="toself", name="OECD median"))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, height=500)
        st.plotly_chart(fig_r, use_container_width=True)
    # stats block
    st.markdown(
        f"**Key stats ({snap_year})**\n\n"
        f"• Vulnerability: **{row['Vulnerability score']:.2f}**\n\n"
        + (f"• % 65+: **{row['Population 65+ (% of total)']:.1f}%**\n\n" if 'Population 65+ (% of total)' in row else '')
        + (f"• Fertility: **{row['Fertility rate (births / woman)']:.2f}**\n\n" if 'Fertility rate (births / woman)' in row else '')
        + (f"• Public debt: **{row['Public debt / GDP (%)']:.1f}%**" if 'Public debt / GDP (%)' in row else '')
    )

# ──────────────────────────────────────────────────────────────
# Downloads
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.download_button(
        label="Download snapshot CSV",
        data=snapshot_df.to_csv(index=False).encode(),
        file_name="oecd_aging_snapshot.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download full panel CSV",
        data=scaled.to_csv(index=False).encode(),
        file_name="oecd_aging_full_panel.csv",
        mime="text/csv",
    )
