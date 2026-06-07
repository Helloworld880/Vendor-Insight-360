"""Tests for the real analytics modules (churn, cohorts, forecasting, stats, segments).

These run against the actual demo CSVs in `Data layer/` — the same data
the dashboard serves — so they double as data-contract tests.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core_modules.analytics_utils import normalize_quarter
from core_modules.churn_model import FEATURES, ChurnPredictor
from core_modules.cohort_analysis import (
    cohort_retention_matrix,
    quarterly_retention,
    renewal_funnel,
)
from core_modules.forecasting import forecast_scores
from core_modules.stats_tests import run_all_insights
from core_modules.vendor_clustering import segment_vendors

DATA = ROOT / "Data layer"


@pytest.fixture(scope="module")
def outcomes() -> pd.DataFrame:
    return pd.read_csv(DATA / "vendor_outcomes.csv")


@pytest.fixture(scope="module")
def performance() -> pd.DataFrame:
    df = pd.read_csv(DATA / "performance.csv")
    return df.rename(
        columns={"on_time_delivery": "on_time_pct", "defect_rate": "defect_rate_pct"}
    )


@pytest.fixture(scope="module")
def financial() -> pd.DataFrame:
    return pd.read_csv(DATA / "financial_metrics.csv")


@pytest.fixture(scope="module")
def risk() -> pd.DataFrame:
    return pd.read_csv(DATA / "risk_history.csv")


@pytest.fixture(scope="module")
def vendors() -> pd.DataFrame:
    return pd.read_csv(DATA / "vendors.csv").rename(columns={"vendor_id": "id"})


# ── analytics_utils ──────────────────────────────────────────────────────
def test_normalize_quarter_handles_both_formats():
    assert normalize_quarter("2024-Q1") == pd.Period("2024Q1", freq="Q")
    assert normalize_quarter("Q1-2024") == pd.Period("2024Q1", freq="Q")


# ── churn model ──────────────────────────────────────────────────────────
class TestChurnModel:
    @pytest.fixture(scope="class")
    def trained(self, outcomes, performance, financial, risk) -> ChurnPredictor:
        model = ChurnPredictor()
        model.train(outcomes, performance, financial, risk)
        return model

    def test_panel_has_no_target_leakage(self, outcomes, performance, financial, risk):
        panel = ChurnPredictor.build_panel(outcomes, performance, financial, risk)
        last_q = panel.groupby("vendor_id")["quarter"].transform("max")
        # The final quarter per vendor has no next-quarter label.
        assert panel.loc[panel["quarter"] == last_q, "churn_next"].isna().all()

    def test_metrics_are_reported(self, trained):
        m = trained.metrics
        assert m is not None
        assert 0 <= m.roc_auc <= 1
        assert m.n_train > m.n_test > 0
        assert 0 < m.base_churn_rate < 0.2  # churn is a rare event

    def test_predictions_are_probabilities(self, trained, vendors):
        scored = trained.predict_current(vendors)
        assert scored["churn_probability"].between(0, 1).all()
        assert "value_at_risk" in scored.columns
        # sorted descending so the riskiest vendor leads the table
        assert scored["churn_probability"].is_monotonic_decreasing

    def test_feature_importance_covers_all_features(self, trained):
        imp = trained.feature_importance()
        assert set(imp["feature"]) == set(FEATURES)
        assert (imp["importance"] >= 0).all()


# ── cohort analysis ──────────────────────────────────────────────────────
class TestCohorts:
    def test_retention_matrix_shape_and_bounds(self, outcomes, performance):
        matrix = cohort_retention_matrix(outcomes, performance)
        assert matrix.shape[0] == 4  # quartile cohorts
        values = matrix.to_numpy()
        assert np.nanmin(values) >= 0 and np.nanmax(values) <= 100

    def test_quarterly_retention_survival_is_decreasing(self, outcomes):
        ret = quarterly_retention(outcomes)
        survival = ret["cumulative_survival_pct"].to_numpy()
        assert (np.diff(survival) <= 0).all()

    def test_funnel_is_monotonic(self, vendors, outcomes):
        funnel = renewal_funnel(vendors, outcomes)
        counts = funnel["vendors"].to_numpy()
        assert (np.diff(counts) <= 0).all()
        assert counts[0] == 120


# ── forecasting ──────────────────────────────────────────────────────────
class TestForecasting:
    def test_portfolio_forecast_has_backtest(self, performance):
        result = forecast_scores(performance, horizon=6)
        assert len(result.forecast) == 6
        assert result.backtest_points > 0
        assert result.mape_model > 0 and result.mape_naive > 0
        # Prediction interval brackets the point forecast.
        assert (result.forecast["lower"] <= result.forecast["predicted_score"]).all()
        assert (result.forecast["upper"] >= result.forecast["predicted_score"]).all()

    def test_vendor_forecast_clips_to_score_range(self, performance):
        vendor = performance["vendor_name"].iloc[0]
        result = forecast_scores(performance, vendor_name=vendor, horizon=6)
        assert result.forecast["predicted_score"].between(0, 100).all()

    def test_too_little_history_raises(self, performance):
        tiny = performance.head(3)
        with pytest.raises(ValueError, match="at least 6"):
            forecast_scores(tiny)


# ── statistical insights ─────────────────────────────────────────────────
class TestStats:
    def test_full_battery_runs(self, outcomes, performance, financial):
        insights = run_all_insights(outcomes, performance, financial)
        assert len(insights) == 5
        for ins in insights:
            assert 0 <= ins.p_value <= 1
            assert ins.effect_name
            assert ins.interpretation

    def test_churned_vendors_underperform(self, outcomes, performance):
        insights = run_all_insights(outcomes, performance, pd.DataFrame())
        churn_test = next(i for i in insights if "churn" in i.question.lower())
        # Direction check: churned vendors should score lower (negative t / d).
        assert churn_test.statistic < 0


# ── segmentation ─────────────────────────────────────────────────────────
class TestSegmentation:
    def test_segments_are_named_and_scored(self, vendors, performance, financial):
        perf_agg = (
            performance.groupby("vendor_id", as_index=False)
            .agg(
                avg_performance=("overall_score", "mean"),
                avg_on_time=("on_time_pct", "mean"),
                avg_defect_rate=("defect_rate_pct", "mean"),
            )
        )
        vwp = vendors.merge(perf_agg, left_on="id", right_on="vendor_id")
        result = segment_vendors(vwp, financial)
        assert 2 <= result.k <= 6
        assert -1 <= result.silhouette <= 1
        assert result.segments["segment"].notna().all()
        assert len(result.profile) == result.k
