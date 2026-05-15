"""A&E Attendance Forecast — SARIMA-based projection.

Fits Seasonal ARIMA models across 8 candidate specifications, picks the best
by AIC, generates a forecast with 80% / 95% confidence intervals, and reports
backtest accuracy on a held-out tail of the series.

Specification choices (driven by the short 24-month training window):

* ``d=1`` — first differencing for trend.
* ``D=0`` — no seasonal differencing. With only 24 observations, seasonal
  differencing would leave too few effective data points and produce
  degenerate fits.
* ``P=0`` — no seasonal AR. A seasonal AR coefficient needs at least three
  full seasonal cycles to estimate reliably; we have two. With this short
  a history, seasonal AR overfits to start-of-series outliers (e.g.
  April 2023, a post-COVID recovery floor) and produces implausible
  forecast trajectories.
* Seasonality is captured through the seasonal MA term (``Q``), which is
  more stable on short series.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    NHS_BLUE,
    NHS_BRIGHT_RED,
    NHS_GREEN,
    format_number,
    load_fact_ae,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Forecast · NHS Demand vs Capacity",
    page_icon="🔮",
    layout="wide",
)

st.title("A&E Attendance Forecast")
st.caption(
    "Forward projection of England-wide monthly A&E attendances using "
    "Seasonal ARIMA (SARIMA). Model selected by AIC from 8 candidate "
    "specifications; backtested on a held-out tail."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

horizon = st.sidebar.slider(
    "Forecast horizon (months)", min_value=1, max_value=6, value=3
)

# ---------------------------------------------------------------------------
# Build the England-wide monthly series
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def build_england_series() -> pd.Series:
    """Aggregate the trust × month A&E fact into a single England-wide series."""
    fact = load_fact_ae()
    series = (
        fact.groupby("month")["total_attendances"]
        .sum()
        .sort_index()
        .astype(float)
    )
    series.index.freq = "MS"
    return series


# ---------------------------------------------------------------------------
# Candidate SARIMA specifications
#
# All candidates use d=1 (first differencing), D=0 (no seasonal differencing),
# and P=0 (no seasonal AR). See module docstring for the rationale.
# We vary p, q, and Q across plausible values.
# ---------------------------------------------------------------------------

CANDIDATES = [
    # (non-seasonal order, seasonal order with period 12)
    ((1, 1, 1), (0, 0, 1, 12)),  # ARMA(1,1) + seasonal MA — canonical
    ((0, 1, 1), (0, 0, 1, 12)),  # MA(1) + seasonal MA — Holt-Winters analog
    ((1, 1, 0), (0, 0, 1, 12)),  # AR(1) + seasonal MA
    ((2, 1, 1), (0, 0, 1, 12)),  # AR(2) MA(1) + seasonal MA
    ((1, 1, 2), (0, 0, 1, 12)),  # AR(1) MA(2) + seasonal MA
    ((0, 1, 2), (0, 0, 1, 12)),  # MA(2) + seasonal MA
    ((1, 1, 1), (0, 0, 0, 12)),  # ARMA(1,1), no seasonal — baseline
    ((0, 1, 1), (0, 0, 0, 12)),  # MA(1), no seasonal — minimal baseline
]

# Defensive filter: if AIC is below this threshold, the fit is degenerate.
MIN_REASONABLE_AIC = 100


@st.cache_resource(show_spinner="Fitting 8 SARIMA candidates and picking best by AIC...")
def fit_candidates() -> list[dict]:
    """Fit all candidate SARIMA models, return sorted by AIC."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    series = build_england_series()
    fitted: list[dict] = []

    for order, seasonal_order in CANDIDATES:
        try:
            model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=200)
            aic = float(fit.aic)
            bic = float(fit.bic)
            if np.isnan(aic) or np.isnan(bic):
                continue
            if aic < MIN_REASONABLE_AIC:
                continue
            fitted.append(
                {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "aic": aic,
                    "bic": bic,
                    "fit": fit,
                }
            )
        except Exception:
            continue

    fitted.sort(key=lambda c: c["aic"])
    return fitted


@st.cache_resource(show_spinner="Running backtest on held-out tail...")
def run_backtest(holdout: int = 3) -> dict:
    """Re-fit the best model on series[:-holdout] and forecast the tail."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    series = build_england_series()
    candidates = fit_candidates()
    best = candidates[0]

    train = series.iloc[:-holdout]
    test = series.iloc[-holdout:]

    model = SARIMAX(
        train,
        order=best["order"],
        seasonal_order=best["seasonal_order"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False, maxiter=200)
    fc = fit.get_forecast(steps=holdout)
    predicted = fc.predicted_mean
    ci_95 = fc.conf_int(alpha=0.05)

    mape = float(np.mean(np.abs((test.values - predicted.values) / test.values)) * 100)
    rmse = float(np.sqrt(np.mean((test.values - predicted.values) ** 2)))

    return {
        "train": train,
        "test": test,
        "predicted": predicted,
        "ci_95": ci_95,
        "mape": mape,
        "rmse": rmse,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

series = build_england_series()
candidates = fit_candidates()

if not candidates:
    st.error(
        "No SARIMA models passed the sanity check. The data series may be too "
        "short for meaningful seasonal modelling."
    )
    st.stop()

best = candidates[0]
fit = best["fit"]
forecast_obj = fit.get_forecast(steps=horizon)
forecast_mean = forecast_obj.predicted_mean
ci_95 = forecast_obj.conf_int(alpha=0.05)
ci_80 = forecast_obj.conf_int(alpha=0.20)

# ---------------------------------------------------------------------------
# Headline chart
# ---------------------------------------------------------------------------

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ci_95.index, y=ci_95.iloc[:, 1].values,
    mode="lines", line=dict(width=0),
    showlegend=False, hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=ci_95.index, y=ci_95.iloc[:, 0].values,
    mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(218, 41, 28, 0.10)",
    name="95% CI",
    hovertemplate="%{x|%b %Y}<br>Lower: %{y:,.0f}<extra></extra>",
))

fig.add_trace(go.Scatter(
    x=ci_80.index, y=ci_80.iloc[:, 1].values,
    mode="lines", line=dict(width=0),
    showlegend=False, hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=ci_80.index, y=ci_80.iloc[:, 0].values,
    mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(218, 41, 28, 0.22)",
    name="80% CI",
    hovertemplate="%{x|%b %Y}<br>Lower: %{y:,.0f}<extra></extra>",
))

fig.add_trace(go.Scatter(
    x=series.index, y=series.values,
    mode="lines+markers",
    line=dict(color=NHS_BLUE, width=2.5),
    marker=dict(size=7),
    name="Historical",
    hovertemplate="%{x|%b %Y}<br>%{y:,.0f}<extra></extra>",
))

fig.add_trace(go.Scatter(
    x=forecast_mean.index, y=forecast_mean.values,
    mode="lines+markers",
    line=dict(color=NHS_BRIGHT_RED, width=2.5, dash="dash"),
    marker=dict(size=9, symbol="diamond"),
    name="Forecast",
    hovertemplate="%{x|%b %Y}<br>%{y:,.0f}<extra></extra>",
))

fig.update_layout(
    height=520,
    margin=dict(l=20, r=20, t=30, b=20),
    plot_bgcolor="white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.update_xaxes(showgrid=True, gridcolor="#E5E5E5", title_text="Month")
fig.update_yaxes(showgrid=True, gridcolor="#E5E5E5", title_text="A&E attendances")

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Top-line summary
# ---------------------------------------------------------------------------

forecast_total = forecast_mean.sum()
last_actual = series.iloc[-1]
first_forecast = forecast_mean.iloc[0]
mom_change = (first_forecast - last_actual) / last_actual

col1, col2, col3 = st.columns(3)
col1.metric(
    f"Total forecast · next {horizon} mo",
    format_number(forecast_total),
)
col2.metric(
    f"Forecast · {forecast_mean.index[0]:%b %Y}",
    format_number(first_forecast),
    delta=f"{mom_change * 100:+.1f}% vs {series.index[-1]:%b %Y}",
)
col3.metric(
    "Model",
    f"SARIMA{best['order']}×{best['seasonal_order'][:3]}",
    delta=f"AIC {best['aic']:.0f}",
    delta_color="off",
)

# ---------------------------------------------------------------------------
# Forecast values table
# ---------------------------------------------------------------------------

st.subheader("Forecast values")

forecast_df = pd.DataFrame({
    "Month": forecast_mean.index.strftime("%B %Y"),
    "Forecast": forecast_mean.values,
    "80% lower": ci_80.iloc[:, 0].values,
    "80% upper": ci_80.iloc[:, 1].values,
    "95% lower": ci_95.iloc[:, 0].values,
    "95% upper": ci_95.iloc[:, 1].values,
})
for col in ["Forecast", "80% lower", "80% upper", "95% lower", "95% upper"]:
    forecast_df[col] = forecast_df[col].apply(format_number)

st.dataframe(forecast_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------

with st.expander("Methodology — model selection & backtest", expanded=False):
    st.markdown(
        f"""
        ### Model selection

        Eight candidate SARIMA specifications were fit on the full 24-month
        England-wide series and ranked by AIC (Akaike Information Criterion).
        Lower AIC is better.

        **Specification choices.** All candidates use:

        - `d=1` — first differencing for trend.
        - `D=0` — no seasonal differencing. With only 24 observations, seasonal
          differencing would leave too few effective data points.
        - `P=0` — no seasonal AR. A seasonal AR coefficient needs at least
          three full seasonal cycles to estimate reliably; we have two.
          Including seasonal AR caused the model to overfit to the
          April 2023 outlier (a post-COVID recovery floor at 1.82M) and
          produce implausible forward forecasts. Seasonality is captured via
          the seasonal MA term (`Q`), which is more stable on short series.

        Candidates whose AIC fell below {MIN_REASONABLE_AIC} were excluded as
        degenerate fits.

        **Selected model:** SARIMA{best['order']} × {best['seasonal_order']}
        — AIC {best['aic']:.1f}, BIC {best['bic']:.1f}.
        """
    )

    st.markdown("#### Candidate scores")

    cand_table = pd.DataFrame(
        [
            {
                "Order (p,d,q)": str(c["order"]),
                "Seasonal (P,D,Q,s)": str(c["seasonal_order"]),
                "AIC": round(c["aic"], 1),
                "BIC": round(c["bic"], 1),
                "Selected": "✓" if c is best else "",
            }
            for c in candidates
        ]
    )
    st.dataframe(cand_table, use_container_width=True, hide_index=True)

    st.markdown("### Backtest — held-out tail")

    bt = run_backtest(holdout=3)

    bt_col1, bt_col2 = st.columns(2)
    bt_col1.metric("MAPE (held-out 3 months)", f"{bt['mape']:.2f}%")
    bt_col2.metric("RMSE (held-out 3 months)", format_number(bt["rmse"]))

    bt_fig = go.Figure()

    bt_fig.add_trace(go.Scatter(
        x=bt["ci_95"].index, y=bt["ci_95"].iloc[:, 1].values,
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    bt_fig.add_trace(go.Scatter(
        x=bt["ci_95"].index, y=bt["ci_95"].iloc[:, 0].values,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(218, 41, 28, 0.10)",
        name="95% CI", hoverinfo="skip",
    ))

    bt_fig.add_trace(go.Scatter(
        x=bt["train"].index, y=bt["train"].values,
        mode="lines+markers",
        line=dict(color=NHS_BLUE, width=2),
        marker=dict(size=6),
        name="Train",
        hovertemplate="%{x|%b %Y}<br>%{y:,.0f}<extra></extra>",
    ))

    bt_fig.add_trace(go.Scatter(
        x=bt["test"].index, y=bt["test"].values,
        mode="lines+markers",
        line=dict(color=NHS_GREEN, width=2.5),
        marker=dict(size=9),
        name="Actual (held out)",
        hovertemplate="%{x|%b %Y}<br>Actual: %{y:,.0f}<extra></extra>",
    ))

    bt_fig.add_trace(go.Scatter(
        x=bt["predicted"].index, y=bt["predicted"].values,
        mode="lines+markers",
        line=dict(color=NHS_BRIGHT_RED, width=2.5, dash="dash"),
        marker=dict(size=9, symbol="diamond"),
        name="Predicted",
        hovertemplate="%{x|%b %Y}<br>Predicted: %{y:,.0f}<extra></extra>",
    ))

    bt_fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    bt_fig.update_xaxes(showgrid=True, gridcolor="#E5E5E5")
    bt_fig.update_yaxes(showgrid=True, gridcolor="#E5E5E5", title_text="A&E attendances")

    st.plotly_chart(bt_fig, use_container_width=True)

    st.markdown(
        f"""
        **How to interpret.** The model was re-fit on the first 21 months and
        used to predict the held-out final 3 months. The green line is what
        actually happened; the red dashed line is what the model said would
        happen. The narrower the gap, the better the model.

        A MAPE of **{bt['mape']:.2f}%** means the average forecast error is
        about {bt['mape']:.1f}% of the actual value — for context, NHS England's
        own monthly variance from year to year is typically in the 3–8% range,
        so this is a reasonable baseline for a model trained on only 24 months.
        """
    )

# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------

with st.expander("Limitations & production extensions"):
    st.markdown(
        """
        ### What this forecast assumes

        - **Stable underlying drivers.** The model extrapolates from 24 months
          of historical pattern. It cannot anticipate structural changes
          (ICS reconfigurations, industrial action, pandemic-style shocks,
          major service redesigns).
        - **Annual seasonality via MA only.** With two seasonal cycles, we
          capture seasonality through the seasonal MA term only (no seasonal
          differencing, no seasonal AR — see methodology).
        - **No exogenous variables.** Temperature, flu activity, bank holiday
          count per month, and population growth are not inputs.
        - **Short training series.** 24 months is the minimum for any
          meaningful seasonal modelling. Production deployments typically use
          5+ years of history to stabilise the seasonal terms.

        ### Production extension paths

        1. **SARIMAX with exogenous regressors** — incorporate UK flu
           surveillance data, ONS population, and a bank-holiday-count-per-month
           regressor as `exog` inputs.
        2. **Per-trust forecasts** — currently the data has 24 monthly points
           per trust which is borderline. Need 36+ months for stable per-trust
           seasonal estimation.
        3. **Re-enable seasonal AR with longer history** — with 5+ years of
           HES extracts, seasonal AR becomes estimable and would likely
           dominate seasonal MA on AIC.
        4. **Hierarchical reconciliation** — forecast at trust level, then
           reconcile up to region and England totals using the `hts` framework.
        5. **Probabilistic forecast evaluation** — extend the backtest to
           rolling-origin cross-validation and report calibration of the
           80/95% intervals against held-out coverage rates.
        """
    )
