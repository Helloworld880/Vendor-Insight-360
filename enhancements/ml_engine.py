import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split


class MLEngine:

    def __init__(self, db):
        self.db = db
        self.model = None

    # ─────────────────────────────
    # Train ML model
    # ─────────────────────────────
    def train_model(self, df):

        features = [
            "on_time_pct",
            "defect_rate_pct",
            "quality_score"
        ]

        target = "overall_score"

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        score = self.model.score(X_test, y_test)

        return score

    # ─────────────────────────────
    # Predict vendor performance
    # ─────────────────────────────
    def predict(self, data):

        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict(data)

    # ─────────────────────────────
    # Vendor Risk Prediction
    # ─────────────────────────────
    def predict_vendor_risks(self):

        df = self.db.get_vendors_with_performance()

        if df.empty:
            return pd.DataFrame()

        df["overall_risk"] = (
            (100 - df["avg_performance"]) * 0.5 +
            df["avg_defect_rate"] * 5 +
            (100 - df["avg_on_time"]) * 0.3
        )

        def label(score):
            if score > 60:
                return "High"
            elif score > 35:
                return "Medium"
            else:
                return "Low"

        df["ml_risk_label"] = df["overall_risk"].apply(label)

        return df

    # NOTE: churn prediction and performance forecasting moved to
    # core_modules/churn_model.py (supervised, leakage-safe) and
    # core_modules/forecasting.py (backtested Holt-Winters). The old
    # heuristics that lived here were rule-based and have been removed.

    # ─────────────────────────────
    # Anomaly Detection
    # ─────────────────────────────
    def detect_anomalies(self):

        df = self.db.get_vendors_with_performance()

        if df.empty:
            return pd.DataFrame()

        features = df[
            ["avg_performance", "avg_on_time", "avg_quality", "avg_defect_rate"]
        ]

        model = IsolationForest(contamination=0.1)

        df["anomaly_score"] = model.fit_predict(features)

        df["is_anomaly"] = df["anomaly_score"] == -1

        return df

    # ─────────────────────────────
    # Retrain models
    # ─────────────────────────────
    def retrain(self):

        perf = self.db.get_performance_data()

        if perf.empty:
            return

        self.train_model(perf)