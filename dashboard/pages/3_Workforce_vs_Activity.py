"""Workforce vs Activity — the demand-vs-capacity scatter, interactive."""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    NHS_BLUE,
    format_number,
    load_dim_trust,
    load_fact_ae,
    load_fact_workforce,
)

st.set_page_config(
    page_title="Workforce vs Activity · NHS Demand vs Capacity",
    page_icon="👥",
    layout="wide",
)

st.title("Workforce vs Activity")
st.caption(
    "The demand-vs-capacity question: how much A&E activity does each trust "
    "handle, given the size of its workforce?"
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

dim_trust = load_dim_trust()
fact_ae = load_fact_ae()
fact_wf = load_fact_workforce()

# Sidebar filters
all_regions = sorted(dim_trust["nhse_region"].unique())
selected_regions = st.sidebar.multiselect(
    "NHS England regions", all_regions, default=all_regions
)
if not selected_regions:
    st.warning("Select at least one region in the sidebar.")
    st.stop()

# Month picker — default to latest A&E month
all_ae_months = sorted(fact_ae["month"].unique(), reverse=True)
month_labels = [pd.Timestamp(m).strftime("%B %Y") for m in all_ae_months]
selected_label = st.sidebar.selectbox("A&E snapshot month", month_labels, index=0)
selected_month = all_ae_months[month_labels.index(selected_label)]

# ---------------------------------------------------------------------------
# Build the merged snapshot
# ---------------------------------------------------------------------------

# Latest workforce snapshot (use latest available month — workforce is slow-moving)
wf_latest_month = fact_wf["month"].max()
wf_totals = fact_wf[
    (fact_wf["month"] == wf_latest_month)
    & (fact_wf["main_staff_group"] == "All staff groups")
    & (fact_wf["staff_group_1"] == "All staff groups")
][["org_code", "fte", "headcount"]]

ae_snapshot = fact_ae[fact_ae["month"] == selected_month][
    ["org_code", "total_attendances", "four_hour_performance"]
]

merged = (
    ae_snapshot.merge(wf_totals, on="org_code")
    .merge(dim_trust[["org_code", "org_name", "nhse_region", "ics"]], on="org_code")
)
merged = merged[merged["nhse_region"].isin(selected_regions)]
merged = merged[merged["total_attendances"] > 0].copy()
merged["attendances_per_fte"] = merged["total_attendances"] / merged["fte"]
merged["org_name_title"] = merged["org_name"].str.title()
merged["four_hour_perf_pct"] = merged["four_hour_performance"] * 100

# ---------------------------------------------------------------------------
# Headline scatter
# ---------------------------------------------------------------------------

st.markdown(
    f"**{len(merged)} trusts** · A&E month: **{pd.Timestamp(selected_month).strftime('%B %Y')}** · "
    f"Workforce snapshot: **{pd.Timestamp(wf_latest_month).strftime('%B %Y')}**"
)

fig = px.scatter(
    merged,
    x="fte",
    y="total_attendances",
    color="four_hour_perf_pct",
    color_continuous_scale="RdYlGn",
    range_color=(60, 100),
    size="total_attendances",
    size_max=28,
    hover_name="org_name_title",
    hover_data={
        "fte": ":,.0f",
        "total_attendances": ":,.0f",
        "four_hour_perf_pct": ":.1f",
        "nhse_region": True,
        "ics": True,
        "org_name_title": False,
        "four_hour_performance": False,
    },
    labels={
        "fte": "Total trust workforce (FTE)",
        "total_attendances": "Monthly A&E attendances",
        "four_hour_perf_pct": "4-hr performance (%)",
        "nhse_region": "Region",
        "ics": "ICS",
    },
)
fig.update_layout(
    height=620,
    margin=dict(l=20, r=20, t=20, b=20),
    plot_bgcolor="white",
    coloraxis_colorbar=dict(title="4-hr perf<br>(%)", thickness=15),
)
fig.update_xaxes(showgrid=True, gridcolor="#E5E5E5")
fig.update_yaxes(showgrid=True, gridcolor="#E5E5E5")

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    **How to read this chart.** Each dot is one NHS Trust in the selected
    snapshot month. The x-axis is total trust workforce (FTE — a proxy for
    operational capacity); the y-axis is A&E attendances (raw demand). Colour
    encodes 4-hour performance — red trusts are struggling, green trusts are
    meeting the standard. The general pattern is a positive correlation
    (bigger trusts handle more activity), but the *deviations* are where the
    story is: trusts below the trend line have lower-than-expected activity
    for their workforce; trusts above are stretched.
    """
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Top / bottom by attendances per FTE
# ---------------------------------------------------------------------------

st.subheader("Productivity proxy: attendances per FTE")
st.caption(
    "Ratio of monthly A&E attendances to total trust FTE. High values mean "
    "the trust handles a lot of A&E demand relative to its workforce — "
    "either lean operations or genuine strain. Use alongside 4-hour "
    "performance for context."
)

col_top, col_bottom = st.columns(2)

with col_top:
    st.markdown("**Highest attendances per FTE**")
    top = (
        merged.nlargest(10, "attendances_per_fte")
        [["org_name_title", "nhse_region", "total_attendances", "fte",
          "attendances_per_fte", "four_hour_perf_pct"]]
    )
    top_display = top.copy()
    top_display["total_attendances"] = top_display["total_attendances"].apply(format_number)
    top_display["fte"] = top_display["fte"].apply(format_number)
    top_display["attendances_per_fte"] = top_display["attendances_per_fte"].apply(lambda x: f"{x:.2f}")
    top_display["four_hour_perf_pct"] = top_display["four_hour_perf_pct"].apply(lambda x: f"{x:.1f}%")
    top_display.columns = ["Trust", "Region", "Attendances", "FTE", "Per FTE", "4-hr"]
    st.dataframe(top_display, use_container_width=True, hide_index=True)

with col_bottom:
    st.markdown("**Lowest attendances per FTE**")
    bottom = (
        merged.nsmallest(10, "attendances_per_fte")
        [["org_name_title", "nhse_region", "total_attendances", "fte",
          "attendances_per_fte", "four_hour_perf_pct"]]
    )
    bottom_display = bottom.copy()
    bottom_display["total_attendances"] = bottom_display["total_attendances"].apply(format_number)
    bottom_display["fte"] = bottom_display["fte"].apply(format_number)
    bottom_display["attendances_per_fte"] = bottom_display["attendances_per_fte"].apply(lambda x: f"{x:.2f}")
    bottom_display["four_hour_perf_pct"] = bottom_display["four_hour_perf_pct"].apply(lambda x: f"{x:.1f}%")
    bottom_display.columns = ["Trust", "Region", "Attendances", "FTE", "Per FTE", "4-hr"]
    st.dataframe(bottom_display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Caveats
# ---------------------------------------------------------------------------

with st.expander("Caveats on this metric"):
    st.markdown(
        """
        **Total trust FTE includes all staff** — not just emergency department
        staff. A trust running a large elective programme, mental health
        services, or community services will have a large total workforce
        relative to A&E activity, pulling its "per FTE" ratio down.

        For a fair ED-specific productivity measure you'd need ECDS-level
        data with ED-specific workforce, which isn't available in the public
        ESR extracts. The ratio here is best read as a **rough operational
        scale indicator**, not a clinical productivity measure.
        """
    )
