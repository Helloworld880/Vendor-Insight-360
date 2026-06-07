"""Time-series forecasting with honest backtesting.

Replaces the previous index-based linear extrapolation with
exponential-smoothing models (statsmodels) evaluated by rolling-origin
backtests. Every forecast is reported alongside the MAPE of a naive
baseline (carry the last value forward) — a forecast only earns trust
by beating that baseline.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from core_modules.analytics_utils import quarter_of_date  # noqa: F401  (shared utils)


@dataclass(frozen=True)
class ForecastResult:
    forecast: pd.DataFrame
    mape_model: float
    mape_naive: float
    method: str
    backtest_points: int
    history: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def beats_naive(self) -> bool:
        return self.mape_model < self.mape_naive


def _monthly_series(performance: pd.DataFrame, vendor_name: str | None = None) -> pd.Series:
    perf = performance.copy()
    if vendor_name is not None:
        perf = perf[perf["vendor_name"] == vendor_name]
    perf["month"] = pd.to_datetime(perf["metric_date"]).dt.to_period("M")
    series = perf.groupby("month")["overall_score"].mean().sort_index()
    series.index = series.index.to_timestamp()
    return series


def _fit(series: pd.Series, seasonal: bool) -> ExponentialSmoothing:
    kwargs = {"trend": "add", "damped_trend": True}
    if seasonal and len(series) >= 24:
        kwargs.update({"seasonal": "add", "seasonal_periods": 12})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ExponentialSmoothing(series, **kwargs).fit(optimized=True)


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _backtest(series: pd.Series, seasonal: bool, folds: int = 6) -> tuple[float, float, int]:
    """Rolling-origin one-step backtest vs the naive last-value forecast."""
    preds, naives, actuals = [], [], []
    for cut in range(len(series) - folds, len(series)):
        train = series.iloc[:cut]
        if len(train) < 6:
            continue
        try:
            model = _fit(train, seasonal)
            preds.append(float(model.forecast(1).iloc[0]))
        except Exception:
            preds.append(float(train.iloc[-1]))
        naives.append(float(train.iloc[-1]))
        actuals.append(float(series.iloc[cut]))

    if not actuals:
        return float("nan"), float("nan"), 0
    return (
        round(_mape(np.array(actuals), np.array(preds)), 2),
        round(_mape(np.array(actuals), np.array(naives)), 2),
        len(actuals),
    )


def forecast_scores(
    performance: pd.DataFrame,
    vendor_name: str | None = None,
    horizon: int = 6,
) -> ForecastResult:
    """Forecast monthly performance scores with a backtested model.

    vendor_name=None forecasts the portfolio average.
    """
    series = _monthly_series(performance, vendor_name)
    if len(series) < 6:
        raise ValueError("Need at least 6 monthly observations to forecast.")

    seasonal = len(series) >= 24
    mape_model, mape_naive, points = _backtest(series, seasonal)

    model = _fit(series, seasonal)
    point = model.forecast(horizon)

    # Approximate prediction interval from in-sample residual spread.
    resid_std = float(np.std(series - model.fittedvalues))
    out = pd.DataFrame(
        {
            "forecast_date": point.index,
            "predicted_score": point.values.clip(0, 100).round(2),
            "lower": (point.values - 1.96 * resid_std).clip(0, 100).round(2),
            "upper": (point.values + 1.96 * resid_std).clip(0, 100).round(2),
        }
    )
    if vendor_name is not None:
        out.insert(0, "vendor_name", vendor_name)

    history = (
        series.rename("actual_score").rename_axis("forecast_date").reset_index()
    )
    method = "Holt-Winters (add. trend, damped" + (", add. seasonality)" if seasonal else ")")
    return ForecastResult(
        forecast=out,
        mape_model=mape_model,
        mape_naive=mape_naive,
        method=method,
        backtest_points=points,
        history=history,
    )
