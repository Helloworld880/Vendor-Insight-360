"""Cohort, retention and funnel analysis over vendor outcomes.

All vendors in the demo dataset share the same contract start date, so
join-date cohorts would be degenerate. Instead vendors are cohorted by
their *initial performance quartile* — the question becomes "do vendors
who start strong stay longer?", which is the business-relevant version.
"""

import numpy as np
import pandas as pd

from core_modules.analytics_utils import normalize_quarter, quarter_of_date

COHORT_LABELS = ["Q1 (weakest start)", "Q2", "Q3", "Q4 (strongest start)"]


def assign_performance_cohorts(
    outcomes: pd.DataFrame, performance: pd.DataFrame
) -> pd.DataFrame:
    """Label each vendor with the quartile of its first-quarter performance."""
    perf = performance.copy()
    perf["quarter"] = quarter_of_date(perf["metric_date"])
    first_q = perf["quarter"].min()
    baseline = (
        perf[perf["quarter"] == first_q]
        .groupby("vendor_id", as_index=False)
        .agg(initial_score=("overall_score", "mean"))
    )
    baseline["cohort"] = pd.qcut(baseline["initial_score"], q=4, labels=COHORT_LABELS)
    return baseline


def cohort_retention_matrix(
    outcomes: pd.DataFrame, performance: pd.DataFrame
) -> pd.DataFrame:
    """Survival rate (share of vendors not yet churned) per cohort per quarter."""
    cohorts = assign_performance_cohorts(outcomes, performance)

    out = outcomes.copy()
    out["quarter"] = out["period"].map(normalize_quarter)
    out = out.merge(cohorts[["vendor_id", "cohort"]], on="vendor_id", how="left")

    # A vendor counts as churned from its first churn quarter onwards.
    churn_q = (
        out[out["churned"] == 1].groupby("vendor_id")["quarter"].min().rename("churn_quarter")
    )
    out = out.merge(churn_q, on="vendor_id", how="left")
    out["survived"] = out["churn_quarter"].isna() | (out["quarter"] < out["churn_quarter"])

    matrix = (
        out.groupby(["cohort", "quarter"], observed=True)["survived"]
        .mean()
        .mul(100)
        .round(1)
        .unstack("quarter")
    )
    matrix.columns = [str(c) for c in matrix.columns]
    return matrix


def quarterly_retention(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Per-quarter churn counts, churn rate and cumulative survival."""
    out = outcomes.copy()
    out["quarter"] = out["period"].map(normalize_quarter)

    by_q = (
        out.groupby("quarter", as_index=False)
        .agg(
            active_vendors=("vendor_id", "nunique"),
            churned=("churned", "sum"),
            renewals=("contract_renewed", "sum"),
        )
        .sort_values("quarter")
    )
    by_q["churn_rate_pct"] = (by_q["churned"] / by_q["active_vendors"] * 100).round(2)
    by_q["cumulative_survival_pct"] = (
        (1 - by_q["churned"] / by_q["active_vendors"]).cumprod() * 100
    ).round(1)
    by_q["quarter"] = by_q["quarter"].astype(str)
    return by_q


def renewal_funnel(vendors: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Monotonic funnel: portfolio → engaged → renewed → loyal → retained."""
    out = outcomes.copy()
    per_vendor = out.groupby("vendor_id").agg(
        renewals=("contract_renewed", "sum"),
        ever_churned=("churned", "max"),
        quarters_observed=("period", "nunique"),
    )

    total = vendors["id"].nunique() if "id" in vendors.columns else len(vendors)
    engaged = len(per_vendor)
    renewed_once = int((per_vendor["renewals"] >= 1).sum())
    loyal = int((per_vendor["renewals"] >= 4).sum())
    retained = int(((per_vendor["renewals"] >= 4) & (per_vendor["ever_churned"] == 0)).sum())

    stages = pd.DataFrame(
        {
            "stage": [
                "All vendors",
                "With outcome history",
                "Renewed ≥ 1 quarter",
                "Renewed ≥ 4 quarters",
                "Loyal & never churned",
            ],
            "vendors": [total, engaged, renewed_once, loyal, retained],
        }
    )
    stages["pct_of_total"] = (stages["vendors"] / max(total, 1) * 100).round(1)
    stages["drop_off"] = (-stages["vendors"].diff().fillna(0)).astype(int)
    return stages
