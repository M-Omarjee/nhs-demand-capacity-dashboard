"""Build clean, processed fact and dimension tables from raw NHS data.

This module is the bridge between raw NHS publications (loaded by ``src.load``)
and the analytical layer the dashboard consumes. It applies the filtering
decisions documented in ``notebooks/01_exploration.ipynb``:

- Keep only NHS Trusts that appear in BOTH the A&E and workforce files
- Drop trusts with negligible A&E activity (mental-health-only and specialist
  trusts that don't run A&E departments)
- Drop the legacy 'Unknown classification' staff-group rows from workforce
- Save results as Parquet for fast, type-preserving access by the dashboard

Three tables are produced, in a classic star schema:

    dim_trust              ← one row per NHS Trust (metadata)
    fact_ae_monthly        ← trust × month A&E activity
    fact_workforce_monthly ← trust × month × staff-group workforce

Run from the repo root:

    python -m src.clean
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.load import load_ae, load_workforce


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum total A&E attendances over the full period for a trust to be
# included in the analytical population. Set generously low — the goal is to
# filter out trusts with structurally zero A&E (specialist orthopaedic /
# cancer / neuroscience / mental-health / community trusts) without
# excluding genuine acute trusts that have occasional reporting gaps.
MIN_TOTAL_ATTENDANCES = 1_000

# Workforce 'main_staff_group' values to retain. We drop legacy
# 'Unknown classification' rows from older data.
VALID_MAIN_STAFF_GROUPS = {
    "All staff groups",
    "Professionally qualified clinical staff",
    "Support to clinical staff",
    "NHS infrastructure support",
}


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def build_dim_trust(ae: pd.DataFrame, wf: pd.DataFrame) -> pd.DataFrame:
    """Build the trust dimension table — one row per NHS Trust.

    The analytical population is the intersection of trusts that appear in
    both raw datasets AND have meaningful A&E activity over the period.
    Metadata (name, region, ICS) is taken from the latest workforce snapshot
    for clean, consistent naming.

    Parameters
    ----------
    ae : pd.DataFrame
        Tidy A&E DataFrame from ``load_ae``.
    wf : pd.DataFrame
        Tidy workforce DataFrame from ``load_workforce``.

    Returns
    -------
    pd.DataFrame
        Columns: ``org_code`` (PK), ``org_name``, ``nhse_region``, ``ics``.
    """
    ae_orgs = set(ae["org_code"].unique())
    wf_orgs = set(wf["org_code"].unique())
    overlap = ae_orgs & wf_orgs

    # Trusts with meaningful A&E activity over the full period
    totals = ae.groupby("org_code")["total_attendances"].sum()
    active = set(totals[totals >= MIN_TOTAL_ATTENDANCES].index)

    keep = overlap & active

    # Pull metadata from the most recent workforce month for clean names
    wf_latest_month = wf["month"].max()
    metadata = (
        wf[(wf["month"] == wf_latest_month) & wf["org_code"].isin(keep)]
        [["org_code", "org_name", "nhse_region", "ics"]]
        .drop_duplicates(subset=["org_code"])
        .sort_values("org_code")
        .reset_index(drop=True)
    )
    return metadata


def build_fact_ae(ae: pd.DataFrame, dim_trust: pd.DataFrame) -> pd.DataFrame:
    """Build the A&E monthly fact table, filtered to the analytical population.

    Parameters
    ----------
    ae : pd.DataFrame
        Tidy A&E DataFrame from ``load_ae``.
    dim_trust : pd.DataFrame
        The trust dimension produced by ``build_dim_trust``.

    Returns
    -------
    pd.DataFrame
        Columns: ``month``, ``org_code``, ``total_attendances``,
        ``total_4hr_breaches``, ``emergency_admissions``, ``dta_waits_4_12h``,
        ``dta_waits_12plus``, ``four_hour_performance``.
    """
    valid_codes = set(dim_trust["org_code"])
    fact = ae[ae["org_code"].isin(valid_codes)].copy()
    fact = fact[
        [
            "month", "org_code", "total_attendances", "total_4hr_breaches",
            "emergency_admissions", "dta_waits_4_12h", "dta_waits_12plus",
            "four_hour_performance",
        ]
    ]
    # four_hour_performance is currently an object dtype because of NA values;
    # coerce to float for downstream maths and Parquet compatibility.
    fact["four_hour_performance"] = pd.to_numeric(
        fact["four_hour_performance"], errors="coerce"
    )
    return fact.sort_values(["org_code", "month"]).reset_index(drop=True)


def build_fact_workforce(wf: pd.DataFrame, dim_trust: pd.DataFrame) -> pd.DataFrame:
    """Build the workforce monthly fact table, filtered to the analytical population.

    Keeps all staff-group rows (both the rollup ``All staff groups`` and the
    granular ``staff_group_1`` breakdowns) — the dashboard layer chooses
    which to query. Drops legacy ``Unknown classification`` rows.

    Parameters
    ----------
    wf : pd.DataFrame
        Tidy workforce DataFrame from ``load_workforce``.
    dim_trust : pd.DataFrame
        The trust dimension produced by ``build_dim_trust``.

    Returns
    -------
    pd.DataFrame
        Columns: ``month``, ``org_code``, ``main_staff_group``,
        ``staff_group_1``, ``fte``, ``headcount``.
    """
    valid_codes = set(dim_trust["org_code"])
    fact = wf[
        wf["org_code"].isin(valid_codes)
        & wf["main_staff_group"].isin(VALID_MAIN_STAFF_GROUPS)
    ].copy()
    fact = fact[
        ["month", "org_code", "main_staff_group", "staff_group_1", "fte", "headcount"]
    ]
    return (
        fact.sort_values(["org_code", "month", "main_staff_group", "staff_group_1"])
            .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def build_all(
    ae_dir: str | Path,
    wf_file: str | Path,
    out_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Run the full clean pipeline and write Parquet outputs to ``out_dir``.

    Parameters
    ----------
    ae_dir : str or Path
        Directory containing raw NHS A&E monthly CSVs.
    wf_file : str or Path
        Path to the Core 1 workforce CSV.
    out_dir : str or Path
        Directory to write Parquet files to. Created if it doesn't exist.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of table name → DataFrame, for in-memory use.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw A&E from {ae_dir}")
    ae = load_ae(ae_dir)
    print(f"  → {len(ae):>7,} rows, {ae['month'].nunique()} months, {ae['org_code'].nunique()} orgs")

    print(f"Loading raw workforce from {Path(wf_file).name}")
    wf = load_workforce(wf_file)
    print(f"  → {len(wf):>7,} rows, {wf['month'].nunique()} months, {wf['org_code'].nunique()} orgs")

    print("\nBuilding tables:")
    dim_trust = build_dim_trust(ae, wf)
    print(f"  dim_trust              → {len(dim_trust):>5,} trusts")

    fact_ae = build_fact_ae(ae, dim_trust)
    print(f"  fact_ae_monthly        → {len(fact_ae):>5,} rows")

    fact_wf = build_fact_workforce(wf, dim_trust)
    print(f"  fact_workforce_monthly → {len(fact_wf):>5,} rows")

    print(f"\nWriting Parquet to {out_dir}/")
    dim_trust.to_parquet(out_dir / "dim_trust.parquet", index=False)
    fact_ae.to_parquet(out_dir / "fact_ae_monthly.parquet", index=False)
    fact_wf.to_parquet(out_dir / "fact_workforce_monthly.parquet", index=False)

    return {
        "dim_trust": dim_trust,
        "fact_ae_monthly": fact_ae,
        "fact_workforce_monthly": fact_wf,
    }


def main() -> None:
    """CLI entry point. Run from repo root: ``python -m src.clean``."""
    parser = argparse.ArgumentParser(
        description="Build cleaned Parquet tables from raw NHS data."
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--ae-dir",
        type=Path,
        default=repo_root / "data" / "raw" / "ae",
        help="Directory containing raw NHS A&E monthly CSVs.",
    )
    parser.add_argument(
        "--wf-file",
        type=Path,
        default=(
            repo_root / "data" / "raw" / "workforce" / "trusts-feb-2026"
            / "Core 1. Staff group - England, NHSE region, ICS and org, Feb-26.csv"
        ),
        help="Path to the Core 1 workforce CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "data" / "processed",
        help="Output directory for Parquet files.",
    )
    args = parser.parse_args()

    build_all(args.ae_dir, args.wf_file, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
