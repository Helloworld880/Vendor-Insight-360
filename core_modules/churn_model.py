"""Real churn prediction trained on labelled vendor outcomes.

Replaces the previous heuristic (``churn_probability = (100 - performance) / 100``)
with a supervised classifier:

- Panel design: one row per (vendor, quarter); the target is whether the
  vendor churns in the *next* quarter, so all features are strictly
  pre-outcome (no leakage).
- Time-aware evaluation: the most recent quarters are held out as a test
  set; cross-validation on the training quarters is grouped by vendor so
  the same vendor never appears in both folds.
- Honest metrics: churn is a rare event (~2% of vendor-quarters), so we
  report ROC-AUC, PR-AUC (average precision) and the Brier score rather
  than accuracy.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core_modules.analytics_utils import normalize_quarter, quarter_of_date

FEATURES = [
    "avg_score",
    "avg_on_time",
    "avg_defect",
    "avg_quality",
    "score_delta",
    "escalation_flag",
    "incident_count",
    "sla_breach_flag",
    "payment_dispute_flag",
    "total_spend",
    "roi_score",
    "overdue_invoices",
    "invoice_accuracy",
    "overall_risk",
]


@dataclass(frozen=True)
class ChurnMetrics:
    model_name: str
    roc_auc: float
    pr_auc: float
    brier: float
    cv_auc_mean: float
    cv_auc_std: float
    n_train: int
    n_test: int
    base_churn_rate: float


class ChurnPredictor:
    """Train and apply a churn classifier on the vendor panel."""

    def __init__(self) -> None:
        self.model: Pipeline | None = None
        self.metrics: ChurnMetrics | None = None
        self._panel: pd.DataFrame | None = None

    # ── Feature engineering ──────────────────────────────────────────────
    @staticmethod
    def build_panel(
        outcomes: pd.DataFrame,
        performance: pd.DataFrame,
        financial: pd.DataFrame,
        risk: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build one row per (vendor, quarter) with next-quarter churn target."""
        out = outcomes.copy()
        out["quarter"] = out["period"].map(normalize_quarter)

        perf = performance.copy()
        perf["quarter"] = quarter_of_date(perf["metric_date"])
        perf_q = (
            perf.groupby(["vendor_id", "quarter"], as_index=False)
            .agg(
                avg_score=("overall_score", "mean"),
                avg_on_time=("on_time_pct", "mean"),
                avg_defect=("defect_rate_pct", "mean"),
                avg_quality=("quality_score", "mean"),
            )
            .sort_values(["vendor_id", "quarter"])
        )
        perf_q["score_delta"] = perf_q.groupby("vendor_id")["avg_score"].diff()

        fin = financial.copy()
        fin["quarter"] = fin["period"].map(normalize_quarter)
        fin_q = fin[
            [
                "vendor_id",
                "quarter",
                "total_spend",
                "roi_score",
                "overdue_invoices",
                "invoice_accuracy",
            ]
        ]

        rk = risk.copy()
        rk["quarter"] = quarter_of_date(rk["assessment_date"])
        risk_q = rk.groupby(["vendor_id", "quarter"], as_index=False).agg(
            overall_risk=("overall_risk", "mean")
        )

        panel = (
            out.merge(perf_q, on=["vendor_id", "quarter"], how="left")
            .merge(fin_q, on=["vendor_id", "quarter"], how="left")
            .merge(risk_q, on=["vendor_id", "quarter"], how="left")
            .sort_values(["vendor_id", "quarter"])
        )

        # Target: churn in the NEXT quarter (features stay pre-outcome).
        panel["churn_next"] = panel.groupby("vendor_id")["churned"].shift(-1)

        # Risk assessments are bi-monthly — forward-fill per vendor.
        panel["overall_risk"] = panel.groupby("vendor_id")["overall_risk"].ffill()
        panel["score_delta"] = panel["score_delta"].fillna(0.0)

        return panel.reset_index(drop=True)

    # ── Training ─────────────────────────────────────────────────────────
    def train(
        self,
        outcomes: pd.DataFrame,
        performance: pd.DataFrame,
        financial: pd.DataFrame,
        risk: pd.DataFrame,
        test_quarters: int = 2,
    ) -> ChurnMetrics:
        panel = self.build_panel(outcomes, performance, financial, risk)
        self._panel = panel

        labelled = panel.dropna(subset=["churn_next"]).copy()
        labelled[FEATURES] = labelled[FEATURES].fillna(labelled[FEATURES].median())

        quarters = sorted(labelled["quarter"].unique())
        cutoff = quarters[-test_quarters]
        train = labelled[labelled["quarter"] < cutoff]
        test = labelled[labelled["quarter"] >= cutoff]

        x_train, y_train = train[FEATURES], train["churn_next"].astype(int)
        x_test, y_test = test[FEATURES], test["churn_next"].astype(int)

        candidates = {
            "logistic_regression": Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            class_weight="balanced", max_iter=2000, random_state=42
                        ),
                    ),
                ]
            ),
            "gradient_boosting": Pipeline(
                [("clf", GradientBoostingClassifier(random_state=42))]
            ),
        }

        groups = train["vendor_id"]
        cv = GroupKFold(n_splits=4)
        scores = {
            name: cross_val_score(
                model, x_train, y_train, cv=cv, groups=groups, scoring="roc_auc"
            )
            for name, model in candidates.items()
        }
        best_name = max(scores, key=lambda n: scores[n].mean())
        best = candidates[best_name]
        best.fit(x_train, y_train)
        self.model = best

        proba = best.predict_proba(x_test)[:, 1]
        self.metrics = ChurnMetrics(
            model_name=best_name,
            roc_auc=round(float(roc_auc_score(y_test, proba)), 3)
            if y_test.nunique() > 1
            else float("nan"),
            pr_auc=round(float(average_precision_score(y_test, proba)), 3)
            if y_test.nunique() > 1
            else float("nan"),
            brier=round(float(brier_score_loss(y_test, proba)), 4),
            cv_auc_mean=round(float(scores[best_name].mean()), 3),
            cv_auc_std=round(float(scores[best_name].std()), 3),
            n_train=len(train),
            n_test=len(test),
            base_churn_rate=round(float(labelled["churn_next"].mean()), 4),
        )
        return self.metrics

    # ── Scoring ──────────────────────────────────────────────────────────
    def predict_current(self, vendors: pd.DataFrame) -> pd.DataFrame:
        """Score the latest quarter: probability each vendor churns next quarter."""
        if self.model is None or self._panel is None:
            raise ValueError("Model not trained yet — call train() first.")

        latest_q = self._panel["quarter"].max()
        current = self._panel[self._panel["quarter"] == latest_q].copy()
        current[FEATURES] = current[FEATURES].fillna(current[FEATURES].median())

        current["churn_probability"] = self.model.predict_proba(current[FEATURES])[:, 1]
        current["churn_risk"] = pd.cut(
            current["churn_probability"],
            bins=[-0.001, 0.05, 0.15, 1.0],
            labels=["Low", "Medium", "High"],
        )

        cols = ["vendor_id", "vendor_name", "churn_probability", "churn_risk"]
        scored = current[cols].copy()

        if not vendors.empty and "contract_value" in vendors.columns:
            value = vendors[["id", "contract_value"]].rename(columns={"id": "vendor_id"})
            scored = scored.merge(value, on="vendor_id", how="left")
            scored["value_at_risk"] = (
                scored["churn_probability"] * scored["contract_value"]
            ).round(0)

        return scored.sort_values("churn_probability", ascending=False).reset_index(
            drop=True
        )

    def feature_importance(self) -> pd.DataFrame:
        """Return per-feature importance for the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet — call train() first.")

        clf = self.model.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importance = clf.feature_importances_
        else:
            importance = np.abs(clf.coef_[0])

        return (
            pd.DataFrame({"feature": FEATURES, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
