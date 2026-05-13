"""Shared utilities for the NHS Demand vs Capacity Streamlit dashboard.

Cached data loaders and brand constants used across all pages.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# Path to the processed Parquet tables. ``__file__`` is dashboard/utils.py,
# so the repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Cached data loaders — read once per session.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dim_trust() -> pd.DataFrame:
    """Load the trust dimension table."""
    return pd.read_parquet(PROCESSED_DIR / "dim_trust.parquet")


@st.cache_data(show_spinner=False)
def load_fact_ae() -> pd.DataFrame:
    """Load the A&E monthly activity fact table."""
    return pd.read_parquet(PROCESSED_DIR / "fact_ae_monthly.parquet")


@st.cache_data(show_spinner=False)
def load_fact_workforce() -> pd.DataFrame:
    """Load the workforce monthly fact table."""
    return pd.read_parquet(PROCESSED_DIR / "fact_workforce_monthly.parquet")


# ---------------------------------------------------------------------------
# NHS brand colors (nhs.uk identity guidelines).
# ---------------------------------------------------------------------------

NHS_BLUE = "#005EB8"
NHS_DARK_BLUE = "#003087"
NHS_BRIGHT_BLUE = "#0072CE"
NHS_LIGHT_BLUE = "#41B6E6"
NHS_BRIGHT_RED = "#DA291C"
NHS_DARK_RED = "#8A1538"
NHS_GREEN = "#009639"
NHS_AQUA_GREEN = "#00A499"
NHS_PURPLE = "#330072"
NHS_PINK = "#AE2573"
NHS_GREY = "#768692"
NHS_DARK_GREY = "#425563"


# ---------------------------------------------------------------------------
# Formatting helpers.
# ---------------------------------------------------------------------------

def format_number(n: float | int) -> str:
    """Format a number with thousands separators, no decimals."""
    return f"{n:,.0f}"


def format_pct(p: float) -> str:
    """Format a 0–1 float as a percentage."""
    return f"{p:.1%}"
