"""Trust Overview — drill into a single NHS Trust's demand and capacity."""

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils import (
    NHS_BLUE,
    NHS_BRIGHT_RED,
    NHS_DARK_GREY,
    NHS_GREEN,
    NHS_GREY,
    format_number,
    load_dim_trust,
    load_fact_ae,
    load_fact_workforce,
)

st.set_page_config(
    page_title="Trust Overview · NHS Demand vs Capacity",
    page_icon="🏥",
    layout="wide",
)

st.title("Trust Overview")
st.caption(
    "Drill into a single NHS Trust's A&E activity, 4-hour performance, "
    "and workforce composition."
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

dim_trust = load_dim_trust()
fact_ae = load_fact_ae()
fact_wf = load_fact_workforce()

# ---------------------------------------------------------------------------
# Trust selector
# ---------------------------------------------------------------------------

trust_names = sorted(dim_trust["org_name"].tolist())
default_trust = (
    "BARTS HEALTH NHS TRUST" if "BARTS HEALTH NHS TRUST" in trust_names else trust_names[0]
)
selected_name = st.selectbox(
    "Select a trust",
    trust_names,
    index=trust_names.index(default_trust),
)
selected_meta = dim_trust[dim_trust["org_name"] == selected_name].iloc[0]
selected_code = selected_meta["org_code"]

st.markdown(
    f"**{selected_meta['org_name'].title()}** &nbsp;·&nbsp; "
    f"`{selected_code}` &nbsp;·&nbsp; "
    f"{selected_meta['nhse_region']} &nbsp;·&nbsp; "
    f"{selected_meta['ics']}"
)

# ---------------------------------------------------------------------------
# Filter facts to this trust
# ---------------------------------------------------------------------------

trust_ae = fact_ae[fact_ae["org_code"] == selected_code].sort_values("month")
trust_wf_total = fact_wf[
    (fact_wf["org_code"] == selected_code)
    & (fact_wf["main_staff_group"] == "All staff groups")
    & (fact_wf["staff_group_1"] == "All staff groups")
].sort_values("month")

if trust_ae.empty:
    st.warning("No A&E data found for this trust.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI row — latest month
# ---------------------------------------------------------------------------

latest = trust_ae.iloc[-1]
latest_wf = trust_wf_total.iloc[-1] if not trust_wf_total.empty else None

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    f"Attendances · {latest['month']:%b %Y}",
    format_number(latest["total_attendances"]),
)
col2.metric(
    "4-hour performance",
    f"{latest['four_hour_performance']:.1%}",
    delta=f"{(latest['four_hour_performance'] - 0.95) * 100:+.1f} pp vs 95%",
    # Default delta_color shows negative deltas in red.
)
col3.metric(
    "Emergency admissions",
    format_number(latest["emergency_admissions"]),
)
if latest_wf is not None:
    col4.metric(
        f"Workforce FTE · {latest_wf['month']:%b %Y}",
        format_number(latest_wf["fte"]),
    )
else:
    col4.metric("Workforce FTE", "—")

st.markdown("---")

# ---------------------------------------------------------------------------
# Time series — trust vs England average
# ---------------------------------------------------------------------------

st.subheader("Time series")

england_perf = (
    fact_ae.groupby("month")
    .apply(
        lambda g: 1 - g["total_4hr_breaches"].sum() / g["total_attendances"].sum(),
        include_groups=False,
    )
    .reset_index(name="england_perf")
)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.10,
    subplot_titles=(
        f"{selected_name.title()} — A&E attendances",
        "4-hour performance — trust vs England",
    ),
)

fig.add_trace(
    go.Scatter(
        x=trust_ae["month"],
        y=trust_ae["total_attendances"],
        mode="lines+markers",
        line=dict(color=NHS_BLUE, width=2.5),
        marker=dict(size=7),
        name="Attendances",
        hovertemplate="%{x|%b %Y}<br>%{y:,.0f}<extra></extra>",
        showlegend=False,
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=trust_ae["month"],
        y=trust_ae["four_hour_performance"] * 100,
        mode="lines+markers",
        line=dict(color=NHS_BRIGHT_RED, width=2.5),
        marker=dict(size=7),
        name="Trust",
        hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra></extra>",
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=england_perf["month"],
        y=england_perf["england_perf"] * 100,
        mode="lines",
        line=dict(color=NHS_GREY, width=1.8, dash="dot"),
        name="England average",
        hovertemplate="%{x|%b %Y}<br>England: %{y:.1f}%<extra></extra>",
    ),
    row=2,
    col=1,
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
    height=620,
    margin=dict(l=20, r=20, t=60, b=20),
    plot_bgcolor="white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=0.45, xanchor="right", x=1),
)
fig.update_xaxes(showgrid=True, gridcolor="#E5E5E5")
fig.update_yaxes(showgrid=True, gridcolor="#E5E5E5")
fig.update_yaxes(title_text="Attendances", row=1, col=1)
fig.update_yaxes(title_text="Performance (%)", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Workforce composition
# ---------------------------------------------------------------------------

if not trust_wf_total.empty:
    st.markdown("---")
    st.subheader("Workforce composition")
    st.caption(f"Snapshot as at {latest_wf['month']:%B %Y}")

    wf_breakdown = fact_wf[
        (fact_wf["org_code"] == selected_code)
        & (fact_wf["month"] == fact_wf["month"].max())
        & (fact_wf["main_staff_group"] != "All staff groups")
        & (fact_wf["staff_group_1"] == "All staff groups")
    ][["main_staff_group", "fte", "headcount"]].sort_values("fte", ascending=False)

    if not wf_breakdown.empty:
        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            colors = [NHS_BLUE, NHS_GREEN, NHS_BRIGHT_RED, NHS_GREY]
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=wf_breakdown["main_staff_group"],
                        values=wf_breakdown["fte"],
                        hole=0.45,
                        marker=dict(colors=colors[: len(wf_breakdown)]),
                        textinfo="label+percent",
                        textposition="outside",
                    )
                ]
            )
            fig_pie.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_table:
            display = wf_breakdown.copy()
            display["fte"] = display["fte"].apply(format_number)
            display["headcount"] = display["headcount"].apply(format_number)
            display.columns = ["Staff group", "FTE", "Headcount"]
            st.dataframe(display, use_container_width=True, hide_index=True)
