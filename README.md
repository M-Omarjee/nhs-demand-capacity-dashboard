# NHS Trust Demand vs Capacity

A trust-level analytics dashboard combining publicly-released NHS A&E activity data with NHS Digital workforce data, with a SARIMA-based short-term forecast layer for A&E attendances.

**🔗 Live demo:** [nhs-demand-capacity.streamlit.app](https://nhs-demand-capacity.streamlit.app)

![Dashboard home page](docs/screenshots/01-home.png)

---

## What this is

NHS England publishes monthly A&E activity. NHS Digital publishes monthly workforce data derived from the Electronic Staff Record (ESR). Neither dataset is hard to download — but joining them, cleaning them into a coherent analytical population, and exposing the result as something a stakeholder can interrogate is non-trivial.

This project does that. It builds a single analytical population of **141 NHS Trusts**, joins their monthly A&E activity (Apr 2023 – Mar 2025) to monthly workforce snapshots, and surfaces:

- Trust-level KPIs vs the 95% 4-hour standard
- Regional demand trends and inter-trust performance heatmaps
- The demand-vs-capacity question: how much A&E activity does each trust handle, given the size of its workforce?
- A short-term SARIMA forecast for England-wide attendances, with a held-out backtest validating the model

The dashboard is built in Python and Streamlit, with Plotly for charts and statsmodels for the forecast.

## Pages

### Home
England-wide headline KPIs (141 trusts, 2.18M attendances in Mar 2025, 73.2% 4-hour performance, 1.07M workforce FTE) and the monthly attendance + performance trend.

### Trust Overview
Drill into a single trust. KPIs, A&E attendance time series vs England average, workforce composition by main staff group.

![Trust Overview](docs/screenshots/02-trust-overview.png)

### Demand Trends
Regional disaggregation. Top-N performance heatmap (red = far below the 95% target, green = meeting it), filterable by region. Year-over-year seasonal view.

![Demand Trends](docs/screenshots/03-demand-trends.png)

### Workforce vs Activity
The core demand-vs-capacity scatter. Each dot is a trust. X-axis: workforce FTE (capacity proxy). Y-axis: monthly A&E attendances (raw demand). Colour: 4-hour performance. Deviations from the trend line surface stretched and slack trusts.

![Workforce vs Activity](docs/screenshots/04-workforce-vs-activity.png)

### Forecast
A SARIMA-based 3-to-6 month forecast of England-wide monthly attendances. Model selected by AIC from 8 candidate specifications, validated by backtest on a held-out tail (MAPE 9.85% on the final 3 months).

![Forecast](docs/screenshots/05-forecast.png)

## Methodology highlights

**Analytical population.** 141 trusts, defined as the intersection of (a) trusts present in the NHS England A&E publication, (b) trusts present in NHS Digital's HCHS workforce data with `DATA_LEVEL='Organisation'`, and (c) a minimum attendance floor of 1,000 per trust-month to exclude small specialist providers whose A&E numbers are not meaningful.

**Data cleaning.** Defensive parsing of NHS England's `Period` field (e.g. `MSitAE-FEBRUARY-2025`), with totals rows dropped. Workforce data filtered to trust-level organisational rows and de-duplicated against an "Unknown classification" staff group. Final star schema: `dim_trust`, `fact_ae_monthly`, `fact_workforce_monthly`.

**SARIMA specification (the interesting part).**

The model deliberately uses `d=1` (first differencing for trend) but `D=0` (no seasonal differencing) and `P=0` (no seasonal AR). The reasoning:

- With only 24 monthly observations, seasonal differencing would leave 12 effective data points — not enough for stable parameter estimation. Several candidate models with `D=1` produced degenerate fits (log-likelihood ≈ 0, AIC ≈ 4) and were excluded.
- A seasonal AR coefficient needs at least three full seasonal cycles to estimate reliably; we have two. Earlier iterations with `P=1` overfit to the April 2023 outlier (a post-COVID recovery floor of 1.82M) and produced implausible forward forecasts — a 16% drop in April 2025, which contradicts the observed Mar→Apr pattern.
- Seasonality is captured through the seasonal MA term (`Q`) instead, which is more stable on short series.

The selected model is **SARIMA(0,1,2) × (0,0,1,12)** — essentially the Holt-Winters analog in SARIMA notation. Backtest MAPE on the held-out final 3 months is **9.85%**.

The dashboard's methodology expander walks the reader through the 8-candidate AIC table and the held-out backtest, so the analytical choices are transparent.

## Data sources

| Source | Dataset | Frequency |
| --- | --- | --- |
| [NHS England](https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/) | A&E attendances, breaches, and 4-hour performance by provider | Monthly |
| [NHS Digital](https://digital.nhs.uk/data-and-information/publications/statistical/nhs-workforce-statistics) | HCHS workforce statistics by trust and staff group (ESR-derived) | Monthly |

Both are publicly released aggregates — the dashboard does not use any patient-level data.

## Tech stack

- **Python 3.13** (Streamlit Cloud deployment) / 3.14 (local dev)
- **pandas + pyarrow** — data ingestion, cleaning, Parquet I/O
- **Streamlit** — multipage web app (`dashboard/app.py` plus four pages under `dashboard/pages/`)
- **Plotly** — interactive charts (time series, heatmaps, scatter, forecast bands)
- **statsmodels** — SARIMA fitting and forecasting via `SARIMAX`

## Limitations

These limitations are surfaced in the dashboard itself but worth restating here.

**HES, not raw HES.** The dashboard uses HES-derived monthly A&E aggregates published by NHS England, not patient-episode-level HES extracts. A production-grade demand-capacity model would query the patient-episode tables directly and aggregate at the analyst's chosen granularity.

**ESR, not raw ESR.** Workforce data comes from NHS Digital's published HCHS aggregates derived from ESR — not the raw ESR system. Vacancy and substantive/bank/agency splits are aggregated.

**24-month training window for the forecast.** Production deployments typically use 5+ years of history to stabilise the seasonal terms; this dashboard uses what's publicly available.

**No exogenous variables.** The forecast does not incorporate flu surveillance, bank holiday counts, ONS population, or weather. These would be `exog` regressors in a SARIMAX extension.

**Trust-level forecasts not included.** Per-trust forecasts with 24 observations would be too noisy to be useful; the dashboard forecasts the England-wide aggregate only.

## Local development

```bash
# Clone
git clone https://github.com/M-Omarjee/nhs-demand-capacity-dashboard.git
cd nhs-demand-capacity-dashboard

# Set up venv (Python 3.13 recommended)
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard/app.py
```

The processed Parquet artifacts (`data/processed/*.parquet`) are committed to the repo so the dashboard runs out of the box. To rebuild them from raw NHS source files, place the downloaded CSVs in `data/raw/ae/` and `data/raw/workforce/` and run:

```bash
python -m src.clean
```

The repo also includes `notebooks/01_exploration.ipynb` with the exploratory data analysis.

## Project structure

```
nhs-demand-capacity-dashboard/
├── README.md
├── requirements.txt
├── runtime.txt                          # Python 3.13 for Streamlit Cloud
├── .streamlit/config.toml               # NHS blue theme
├── src/
│   ├── load.py                          # Raw CSV → DataFrame loaders
│   └── clean.py                         # Build dim/fact tables, persist to Parquet
├── notebooks/
│   └── 01_exploration.ipynb             # EDA narrative
├── data/
│   ├── raw/                             # gitignored — populate locally to rebuild
│   └── processed/                       # committed: dim_trust, fact_ae, fact_workforce
└── dashboard/
    ├── app.py                           # Home page
    ├── utils.py                         # Cached loaders, NHS palette
    └── pages/
        ├── 1_Trust_Overview.py
        ├── 2_Demand_Trends.py
        ├── 3_Workforce_vs_Activity.py
        └── 4_Forecast.py
```

## License

MIT. See `LICENSE`.

---

Built by [Muhammed Omarjee](https://github.com/M-Omarjee) — NHS Foundation Doctor and clinical-AI builder.
