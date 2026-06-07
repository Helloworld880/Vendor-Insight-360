"""Vendor segmentation via K-Means with a defensible choice of k.

Features are standardised before clustering and k is selected by
silhouette score rather than hardcoded. Segments are then profiled and
given business-readable names so the output is actionable, not just
cluster indices.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEGMENT_FEATURES = ["avg_performance", "avg_on_time", "avg_defect_rate", "total_spend"]


@dataclass(frozen=True)
class SegmentationResult:
    segments: pd.DataFrame
    profile: pd.DataFrame
    k: int
    silhouette: float


def _name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Map cluster ids to business personas from their feature profile."""
    names: dict[int, str] = {}
    perf_median = profile["avg_performance"].median()
    spend_median = profile["total_spend"].median()

    for cluster_id, row in profile.iterrows():
        high_perf = row["avg_performance"] >= perf_median
        high_spend = row["total_spend"] >= spend_median
        if high_perf and high_spend:
            names[cluster_id] = "Strategic Partners"
        elif high_perf:
            names[cluster_id] = "Efficient Specialists"
        elif high_spend:
            names[cluster_id] = "Watch List (high spend, low performance)"
        else:
            names[cluster_id] = "Tail Vendors"
    return names


def segment_vendors(
    vendors_with_performance: pd.DataFrame,
    financial: pd.DataFrame,
    k_range: tuple[int, int] = (2, 6),
) -> SegmentationResult:
    """Cluster vendors on performance + spend; pick k by silhouette."""
    spend = (
        financial.groupby("vendor_id", as_index=False)
        .agg(total_spend=("total_spend", "sum"))
    )
    df = vendors_with_performance.merge(spend, on="vendor_id", how="left")

    features = df[SEGMENT_FEATURES].copy()
    features = features.fillna(features.median())
    scaled = StandardScaler().fit_transform(features)

    best_k, best_score, best_labels = 0, -1.0, np.array([])
    for k in range(k_range[0], k_range[1] + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    result = df.copy()
    result["cluster"] = best_labels

    profile = result.groupby("cluster")[SEGMENT_FEATURES].mean().round(2)
    profile["n_vendors"] = result.groupby("cluster").size()
    names = _name_segments(profile)
    result["segment"] = result["cluster"].map(names)
    profile["segment"] = profile.index.map(names)

    return SegmentationResult(
        segments=result,
        profile=profile.reset_index(),
        k=best_k,
        silhouette=round(float(best_score), 3),
    )
