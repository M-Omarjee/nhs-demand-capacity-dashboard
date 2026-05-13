# NHS Trust Demand vs Capacity Dashboard

A trust-level analytics dashboard combining publicly-released NHS Hospital Episode Statistics (HES) activity data with NHS Electronic Staff Record (ESR)-derived workforce data, with a short-term forecast layer for A&E attendances.

Built to demonstrate the kind of demand-and-capacity, intelligent-insights, and predictive-analytics work used in NHS commissioning and consulting contexts.

## Project status

🚧 **In development.** See [project board](#) for current progress.

## Why this exists

NHS acute trusts run on the balance between **demand** (admissions, A&E attendances, outpatient appointments) and **capacity** (workforce, beds, theatre time). This project pulls publicly-available HES-derived and ESR-derived data into a single trust-level view with:

- 12-month activity trends per trust
- 4-hour A&E performance heatmaps
- Staff group composition against activity volume
- 3-month forecast for A&E attendances using SARIMA / Prophet

## Data sources

All datasets are open and publicly downloadable. Refresh cadence and exact endpoints are documented in `src/download.py`.

| Dataset | Source | Granularity | Notes |
|---|---|---|---|
| Monthly A&E Attendances and Admissions | NHS England | Trust, monthly | HES-derived |
| Hospital Admitted Patient Care Activity | NHS Digital / NHS England | Trust, annual | HES-derived |
| NHS Workforce Statistics | NHS Digital | Trust, monthly | ESR-derived |

## Architecture

```
data/raw/        Raw CSV downloads from NHS England / NHS Digital (gitignored)
data/processed/  Cleaned star-schema fact and dimension tables (gitignored)
src/             Data ingestion, cleaning, and schema code
notebooks/       Exploratory analysis and methodology notebooks
dashboard/       Interactive dashboard (Tableau workbook or Streamlit app)
```

## Local setup

```bash
git clone https://github.com/M-Omarjee/nhs-demand-capacity-dashboard.git
cd nhs-demand-capacity-dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

_To be documented as the pipeline is built. The end-state will be:_

```bash
# Download raw data
python -m src.download

# Clean and build schema
python -m src.clean

# Launch dashboard
streamlit run dashboard/app.py
```

## Methodology

_Coming soon — will include star-schema diagram, forecast methodology, and a frank limitations section._

## Limitations

This project uses publicly-released summary extracts of HES- and ESR-derived data, not the patient-episode-level production warehouses (SUS, raw HES, raw ESR), access to which requires NHS Trust roles and DSPT (Data Security and Protection Toolkit) approval.

The dimensional structure and analytical patterns demonstrated here translate directly to production data; the specifics of patient-episode tables would be picked up at deployment.

## Author

**Muhammed Omarjee** — Foundation Doctor & EY alumnus  
[GitHub](https://github.com/M-Omarjee) · London, UK
