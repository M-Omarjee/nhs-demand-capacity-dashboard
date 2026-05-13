"""Demand Trends — regional comparisons, performance heatmap, and seasonal patterns."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    NHS_AQUA_GREEN,
    NHS_BLUE,
    NHS_BRIGHT_BLUE,
    NHS_BRIGHT_RED,
    NHS_DARK_BLUE,
    NHS_GREEN,
    NHS_PINK,
    NHS_PURPLE,
    load_dim_trust,
    load_fact_ae,
)

st.set_page_config(
    page_title="Demand Trends · NHS Demand vs Capacity",
    page_icon="📈",
    layout="wide",
)

st.title("Demand Trends")
st.caption(
    "Regional comparisons, trust-level 4-hour performance heatmap, "
    "and year-over-year seasonal patterns."
)

# ---------------------------------------------------------------------------
# Load + filter
# ---------------------------------------------------------------------------

dim_trust = load_dim_trust()
fact_ae = load_fact_ae()

# Join region info onto A&E facts
ae = fact_ae.merge(
    dim_trust[["org_code", "org_name", "nhse_region"]],
    on="org_code",
)

# Sidebar filter
all_regions = sorted(dim_trust["nhse_region"].unique())
selected_regions = st.sidebar.multiselect(
    "NHS England regions",
    all_regions,
    default=all_regions,
)

if not selected_regions:
    st.warning("Select at least one region in the sidebar.")
    st.stop()

ae_filt = ae[ae["nhse_region"].isin(selected_regions)].copy()

# Region colour palette in a fixed order so the colour is stable across charts
REGION_COLORS = {
    "London": NHS_BLUE,
    "South East": NHS_BRIGHT_BLUE,
    "South West": NHS_AQUA_GREEN,
    "Midlands": NHS_GREEN,
    "East of England": NHS_PURPLE,
    "North West": NHS_PINK,
    "North East and Yorkshire": NHS_DARK_BLUE,
}

# ---------------------------------------------------------------------------
# Chart 1 — attendances by region over time
# ---------------------------------------------------------------------------

st.subheader("Monthly A&E attendances by region")

regional = (
    ae_filt.groupby(["month", "nhse_region"])
    .agg(attendances=("total_attendances", "sum"))
    .reset_index()
)

fig1 = px.line(
    regional,
    x="month",
    y="attendances",
    color="nhse_region",
    color_discrete_map=REGION_COLORS,
    labels={
        "month": "Month",
        "attendances": "Attendances",
        "nhse_region": "Region",
    },
    markers=True,
)
fig1.update_traces(line=dict(width=2.5), marker=dict(size=6))
fig1.update_layout(
    height=460,
    margin=dict(l=20, r=20, t=20, b=20),
    plot_bgcolor="white",
    hovermode="x unified",
    legend=dict(title=""),
)
fig1.update_xaxes(showgrid=True, gridcolor="#E5E5E5")
fig1.update_yaxes(showgrid=True, gridcolor="#E5E5E5")

st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 2 — 4-hour performance heatmap (top N trusts)
# ---------------------------------------------------------------------------

st.subheader("4-hour performance by trust (heatmap)")

top_n = st.sidebar.slider(
    "Heatmap: top N trusts by activity", min_value=10, max_value=50, value=25, step=5
)

st.caption(
    f"Top {top_n} trusts by total A&E activity in the selected regions. "
    "Red = far below 95% target; green = meeting or exceeding."
)

top_codes = (
    ae_filt.groupby("org_code")["total_attendances"].sum().nlargest(top_n).index.tolist()
)
heat = ae_filt[ae_filt["org_code"].isin(top_codes)].copy()
heat["org_name_title"] = heat["org_name"].str.title()

heat_pivot = heat.pivot_table(
    index="org_name_title",
    columns="month",
    values="four_hour_performance",
    aggfunc="mean",
) * 100

# Sort y-axis by mean performance descending (best at top)
heat_pivot = heat_pivot.reindex(
    heat_pivot.mean(axis=1).sort_values(ascending=False).index
)

fig2 = go.Figure(
    data=go.Heatmap(
        z=heat_pivot.values,
        x=heat_pivot.columns,
        y=heat_pivot.index,
        colorscale="RdYlGn",
        zmin=50,
        zmax=100,
        colorbar=dict(title="4-hr perf<br>(%)", thickness=15),
        hovertemplate="<b>%{y}</b><br>%{x|%b %Y}<br>%{z:.1f}%<extra></extra>",
    )
)
fig2.update_layout(
    height=max(450, 22 * top_n),
    margin=dict(l=20, r=20, t=20, b=20),
)
fig2.update_xaxes(tickformat="%b %y", showgrid=False)
fig2.update_yaxes(showgrid=False, tickfont=dict(size=10))

st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 3 — year-over-year seasonal comparison
# ---------------------------------------------------------------------------

st.subheader("Year-over-year — seasonal pattern")
st.caption("Same months stacked across years to make seasonality visible.")

yoy = ae_filt.copy()
yoy["year"] = yoy["month"].dt.year
yoy["month_of_year"] = yoy["month"].dt.month

yoy_totals = (
    yoy.groupby(["year", "month_of_year"])
    .agg(attendances=("total_attendances", "sum"))
    .reset_index()
)

yoy_totals["year"] = yoy_totals["year"].astype(str)

# Year colour palette — recent years bolder
year_colors = {y: c for y, c in zip(
    sorted(yoy_totals["year"].unique()),
    [NHS_BLUE, NHS_BRIGHT_RED, NHS_GREEN, NHS_PURPLE, NHS_AQUA_GREEN],
)}

fig3 = px.line(
    yoy_totals,
    x="month_of_year",
    y="attendances",
    color="year",
    color_discrete_map=year_colors,
    labels={"month_of_year": "Month", "attendances": "Attendances", "year": "Year"},
    markers=True,
)
fig3.update_traces(line=dict(width=2.5), marker=dict(size=7))
fig3.update_xaxes(
    tickmode="array",
    tickvals=list(range(1, 13)),
    ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    showgrid=True,
    gridcolor="#E5E5E5",
)
fig3.update_yaxes(showgrid=True, gridcolor="#E5E5E5")
fig3.update_layout(
    height=400,
    margin=dict(l=20, r=20, t=20, b=20),
    plot_bgcolor="white",
    hovermode="x unified",
    legend=dict(title=""),
)

st.plotly_chart(fig3, use_container_width=True)
