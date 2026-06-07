"""Shared helpers for the analytics modules."""

import pandas as pd


def normalize_quarter(value: str) -> pd.Period:
    """Parse a quarter label into a pandas Period.

    Handles both formats present in the datasets:
    - vendor_outcomes.csv uses "2024-Q1"
    - financial_metrics.csv uses "Q1-2024"
    """
    value = str(value).strip()
    if value.startswith("Q"):
        quarter, year = value.split("-")
        value = f"{year}-{quarter}"
    return pd.Period(value, freq="Q")


def quarter_of_date(dates: pd.Series) -> pd.Series:
    """Map a datetime series to quarterly Periods."""
    return pd.to_datetime(dates).dt.to_period("Q")
