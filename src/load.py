"""Load NHS A&E and workforce CSVs into clean, tidy DataFrames.
 
This module handles all raw-data ingestion. The two public functions are:
 
    load_ae(directory)        → tidy A&E activity DataFrame
    load_workforce(filepath)  → tidy workforce DataFrame
 
Both return data at trust × month grain with consistent column naming, ready
to join on ('org_code', 'month').
"""
 
from pathlib import Path
import warnings
import pandas as pd
 
 
# ---------------------------------------------------------------------------
# A&E loader
# ---------------------------------------------------------------------------
 
# Map textual month names from NHS England's 'Period' field to integers.
MONTH_MAP = {
    "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4,
    "MAY": 5, "JUNE": 6, "JULY": 7, "AUGUST": 8,
    "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12,
}
 
 
def _parse_period(period) -> pd.Timestamp:
    """Convert NHS England's 'Period' string to a first-of-month Timestamp.
 
    Handles the standard format 'MSitAE-FEBRUARY-2025'. Returns ``pd.NaT``
    for anything that doesn't fit — typically totals rows, blank rows, or
    legacy formats from older files. ``load_ae`` then drops those rows.
    """
    if pd.isna(period):
        return pd.NaT
 
    period_str = str(period).strip()
    parts = period_str.split("-")
 
    # Expected: ['MSitAE', 'MONTH', 'YEAR']
    if len(parts) < 3:
        return pd.NaT
 
    try:
        month_name = parts[1].upper()
        year = int(parts[2])
        return pd.Timestamp(year=year, month=MONTH_MAP[month_name], day=1)
    except (KeyError, ValueError):
        return pd.NaT
 
 
def load_ae(directory: str | Path) -> pd.DataFrame:
    """Load all NHS England monthly A&E CSVs in a directory into one tidy frame.
 
    Reads every ``*.csv`` in ``directory``, aggregates the Type 1 / Type 2 /
    Other A&E columns into clean totals, and derives a first-of-month date
    from the 'Period' string. Rows with an unparseable Period (totals rows,
    blank rows, legacy formats) are dropped with a warning.
 
    Parameters
    ----------
    directory : str or Path
        Folder containing one monthly NHS England 'Monthly-AE-*.csv' file
        per month.
 
    Returns
    -------
    pandas.DataFrame
        One row per trust × month with the following columns:
 
        ============================  ============================================
        Column                        Meaning
        ============================  ============================================
        month                         First day of the reporting month (Timestamp)
        org_code                      NHS provider code (joins to workforce)
        org_name                      Provider name as published
        parent_org                    NHS England region (free text)
        total_attendances             Sum of Type 1 + Type 2 + Other attendances
        total_4hr_breaches            Sum of >4hr attendances across all types
        emergency_admissions          Emergency admissions via A&E + other routes
        dta_waits_4_12h               Patients waiting 4–12 hrs from DTA
        dta_waits_12plus              Patients waiting 12+ hrs from DTA
        four_hour_performance         (attendances − breaches) ÷ attendances
        ============================  ============================================
 
    Raises
    ------
    FileNotFoundError
        If no CSV files are present in ``directory``.
    """
    directory = Path(directory)
    csvs = sorted(directory.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No A&E CSV files found in {directory}")
 
    frames = []
    rows_dropped = 0
 
    for csv in csvs:
        df = pd.read_csv(csv)
        df["month"] = df["Period"].apply(_parse_period)
 
        # Drop rows we couldn't date — totals rows, blank rows, legacy formats.
        before = len(df)
        df = df.dropna(subset=["month"]).copy()
        rows_dropped += before - len(df)
 
        # Total demand: sum the three attendance types. We deliberately
        # exclude the 'Booked Appointments' columns — those are a subset
        # of the totals above, not additional attendances.
        df["total_attendances"] = (
            df["A&E attendances Type 1"].fillna(0)
            + df["A&E attendances Type 2"].fillna(0)
            + df["A&E attendances Other A&E Department"].fillna(0)
        )
 
        df["total_4hr_breaches"] = (
            df["Attendances over 4hrs Type 1"].fillna(0)
            + df["Attendances over 4hrs Type 2"].fillna(0)
            + df["Attendances over 4hrs Other Department"].fillna(0)
        )
 
        df["emergency_admissions"] = (
            df["Emergency admissions via A&E - Type 1"].fillna(0)
            + df["Emergency admissions via A&E - Type 2"].fillna(0)
            + df["Emergency admissions via A&E - Other A&E department"].fillna(0)
            + df["Other emergency admissions"].fillna(0)
        )
 
        df["dta_waits_4_12h"] = df[
            "Patients who have waited 4-12 hs from DTA to admission"
        ].fillna(0)
        df["dta_waits_12plus"] = df[
            "Patients who have waited 12+ hrs from DTA to admission"
        ].fillna(0)
 
        # 4-hour performance — guard against divide-by-zero for non-A&E orgs.
        df["four_hour_performance"] = pd.NA
        mask = df["total_attendances"] > 0
        df.loc[mask, "four_hour_performance"] = (
            (df.loc[mask, "total_attendances"] - df.loc[mask, "total_4hr_breaches"])
            / df.loc[mask, "total_attendances"]
        )
 
        df = df.rename(
            columns={
                "Org Code": "org_code",
                "Org name": "org_name",
                "Parent Org": "parent_org",
            }
        )
 
        keep = [
            "month", "org_code", "org_name", "parent_org",
            "total_attendances", "total_4hr_breaches", "emergency_admissions",
            "dta_waits_4_12h", "dta_waits_12plus", "four_hour_performance",
        ]
        frames.append(df[keep])
 
    if rows_dropped:
        warnings.warn(
            f"load_ae: dropped {rows_dropped} rows with unparseable 'Period' "
            "(typically totals rows or blank rows in raw NHS files)."
        )
 
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["org_code", "month"]).reset_index(drop=True)
 
 
# ---------------------------------------------------------------------------
# Workforce loader
# ---------------------------------------------------------------------------
 
VALID_LEVELS = {"National", "NHS England region", "ICS area", "Organisation"}
 
 
def load_workforce(
    filepath: str | Path,
    data_level: str = "Organisation",
) -> pd.DataFrame:
    """Load the NHS HCHS workforce 'Core 1' CSV, filtered to one aggregation level.
 
    The Core 1 file mixes national, regional, ICS and trust-level rows in a
    single long-format file. For trust-level analysis we filter to
    ``data_level='Organisation'``.
 
    Parameters
    ----------
    filepath : str or Path
        Path to 'Core 1. Staff group - England, NHSE region, ICS and org, *.csv'.
    data_level : {'National', 'NHS England region', 'ICS area', 'Organisation'}, default 'Organisation'
        Which aggregation level to retain.
 
    Returns
    -------
    pandas.DataFrame
        One row per org × month × staff_group_1 with columns:
 
        ====================  ===================================================
        Column                Meaning
        ====================  ===================================================
        month                 First day of the reporting month (Timestamp)
        org_code              NHS provider code (joins to A&E)
        org_name              Provider name as published
        nhse_region           NHS England region name
        ics                   Integrated Care System name
        main_staff_group      High-level grouping (e.g. 'Professionally qualified')
        staff_group_1         Detailed staff group (e.g. 'Nurses & health visitors')
        fte                   Full-time equivalent (float)
        headcount             Staff headcount (int)
        ====================  ===================================================
 
    Raises
    ------
    ValueError
        If ``data_level`` is not one of the valid options.
    """
    if data_level not in VALID_LEVELS:
        raise ValueError(
            f"data_level must be one of {VALID_LEVELS}, got {data_level!r}"
        )
 
    df = pd.read_csv(filepath, parse_dates=["DATA_MONTH"])
    df = df[df["DATA_LEVEL"] == data_level].copy()
 
    # Normalise to first-of-month so we can join cleanly against A&E.
    df["month"] = df["DATA_MONTH"].dt.to_period("M").dt.to_timestamp()
 
    df = df.rename(
        columns={
            "ORG_CODE": "org_code",
            "ORG_NAME": "org_name",
            "NHSE_REGION_NAME": "nhse_region",
            "ICS_NAME": "ics",
            "MAIN_STAFF_GROUP": "main_staff_group",
            "STAFF_GROUP_1": "staff_group_1",
            "FTE": "fte",
            "HEADCOUNT": "headcount",
        }
    )
 
    keep = [
        "month", "org_code", "org_name", "nhse_region", "ics",
        "main_staff_group", "staff_group_1", "fte", "headcount",
    ]
    return df[keep].sort_values(["org_code", "month"]).reset_index(drop=True)