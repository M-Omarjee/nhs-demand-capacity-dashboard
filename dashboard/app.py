"""NHS Demand vs Capacity Dashboard — home page.

Run from repo root:
    streamlit run dashboard/app.py
"""

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils import (
    NHS_BLUE,
    NHS_BRIGHT_RED,
    NHS_DARK_GREY,
    format_number,
    load_dim_trust,
    load_fact_ae,
    load_fact_workforce,
)

st.set_page_config(
    page_title="NHS Demand vs Capacity",
    page_icon="🏥",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("NHS Trust Demand vs Capacity")
st.markdown(
    "*A trust-level analytics dashboard combining publicly-released NHS "
    "Hospital Episode Statistics (HES) activity data with NHS Electronic "
    "Staff Record (ESR)-derived workforce data, with a short-term forecast "
    "layer for A&E attendances.*"
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

dim_trust = load_dim_trust()
fact_ae = load_fact_ae()
fact_wf = load_fact_workforce()

# ---------------------------------------------------------------------------
# KPI row — latest month snapshot
# ---------------------------------------------------------------------------

latest_ae_month = fact_ae["month"].max()
latest_ae = fact_ae[fact_ae["month"] == latest_ae_month]

total_attendances = latest_ae["total_attendances"].sum()
total_breaches = latest_ae["total_4hr_breaches"].sum()
england_perf = 1 - (total_breaches / total_attendances)

latest_wf_month = fact_wf["month"].max()
latest_wf = fact_wf[
    (fact_wf["month"] == latest_wf_month)
    & (fact_wf["main_staff_group"] == "All staff groups")
    & (fact_wf["staff_group_1"] == "All staff groups")
]
total_fte = latest_wf["fte"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("NHS Trusts", len(dim_trust))
col2.metric(
    f"A&E attendances · {latest_ae_month:%b %Y}",
    format_number(total_attendances),
)
col3.metric(
    "4-hour performance",
    f"{england_perf:.1%}",
    delta=f"{(england_perf - 0.95) * 100:+.1f} pp vs 95% standard",
    # Default delta_color (`normal`) shows negative deltas in red — correct
    # signal here because below-target performance is bad.
)
col4.metric(
    f"Workforce FTE · {latest_wf_month:%b %Y}",
    format_number(total_fte),
)

st.markdown("---")

# ---------------------------------------------------------------------------
# England-wide trend chart
# ---------------------------------------------------------------------------

st.subheader("England-wide trends")

england_monthly = (
    fact_ae.groupby("month")
    .agg(
        attendances=("total_attendances", "sum"),
        breaches=("total_4hr_breaches", "sum"),
    )
    .reset_index()
)
england_monthly["four_hour_perf"] = (
    1 - england_monthly["breaches"] / england_monthly["attendances"]
)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("Monthly A&E attendances", "4-hour performance"),
)

fig.add_trace(
    go.Scatter(
        x=england_monthly["month"],
        y=england_monthly["attendances"] / 1e6,
        mode="lines+markers",
        line=dict(color=NHS_BLUE, width=2.5),
        marker=dict(size=8),
        name="Attendances (M)",
        hovertemplate="%{x|%b %Y}<br>%{y:.2f} M<extra></extra>",
        showlegend=False,
    ),
    row=1, col=1,
)

fig.add_trace(
    go.Scatter(
        x=england_monthly["month"],
        y=england_monthly["four_hour_perf"] * 100,
        mode="lines+markers",
        line=dict(color=NHS_BRIGHT_RED, width=2.5),
        marker=dict(size=8),
        name="4-hour performance",
        hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra></extra>",
        showlegend=False,
    ),
    row=2, col=1,
)

fig.add_hline(
    y=95,
    line_dash="dash",
    line_color=NHS_DARK_GREY,
    annotation_text="95% standard",
    annotation_position="top right",
    row=2,
    col=1,
)

fig.update_layout(
    height=520,
    margin=dict(l=20, r=20, t=50, b=20),
    plot_bgcolor="white",
    hovermode="x unified",
)
fig.update_xaxes(showgrid=True, gridcolor="#E5E5E5")
fig.update_yaxes(showgrid=True, gridcolor="#E5E5E5")
fig.update_yaxes(title_text="Attendances (millions)", row=1, col=1)
fig.update_yaxes(title_text="Performance (%)", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Navigation hint + methodology note
# ---------------------------------------------------------------------------

st.markdown("---")

col_nav, col_method = st.columns([1, 1])

with col_nav:
    st.markdown(
        """
        ### Navigate the dashboard

        Use the sidebar to explore:

        - **Trust Overview** — drill into a single trust's activity and workforce
        - **Demand Trends** — regional comparisons and seasonal patterns
        - **Workforce vs Activity** — the demand-vs-capacity scatter
        - **Forecast** — 3-month A&E attendance forecast *(coming)*
        """
    )

with col_method:
    st.markdown(
        """
        ### Methodology & limitations

        Data sources: NHS England Monthly A&E Attendances (HES-derived) and
        NHS Digital Workforce Statistics (ESR-derived). The analytical
        population is **141 NHS Trusts** that appear in both datasets and
        have meaningful A&E activity.

        This is **not** the patient-episode-level SUS / raw HES / raw ESR
        warehouse, which requires NHS Trust roles and DSPT approval.
        """
    )
