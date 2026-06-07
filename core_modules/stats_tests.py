"""Statistical hypothesis tests over the vendor datasets.

Each insight reports the test statistic, p-value AND an effect size —
a small p-value with a negligible effect is noise, not news.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


@dataclass(frozen=True)
class InsightResult:
    question: str
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_name: str
    significant: bool
    interpretation: str


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled else 0.0


def _cramers_v(table: pd.DataFrame, chi2: float) -> float:
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n and k else 0.0


def escalation_vs_renewal(outcomes: pd.DataFrame) -> InsightResult:
    """Do quarters with an escalation see fewer contract renewals?"""
    table = pd.crosstab(outcomes["escalation_flag"], outcomes["contract_renewed"])
    chi2, p, _, _ = stats.chi2_contingency(table)
    v = _cramers_v(table, chi2)

    renew_no_esc = outcomes.loc[outcomes["escalation_flag"] == 0, "contract_renewed"].mean()
    renew_esc = outcomes.loc[outcomes["escalation_flag"] == 1, "contract_renewed"].mean()
    return InsightResult(
        question="Do escalations reduce contract renewals?",
        test_name="Chi-squared test of independence",
        statistic=round(float(chi2), 3),
        p_value=round(float(p), 4),
        effect_size=round(v, 3),
        effect_name="Cramér's V",
        significant=p < ALPHA,
        interpretation=(
            f"Renewal rate is {renew_no_esc:.0%} without escalations vs "
            f"{renew_esc:.0%} with escalations "
            f"({'significant' if p < ALPHA else 'not significant'} at α={ALPHA})."
        ),
    )


def performance_vs_churn(outcomes: pd.DataFrame, performance: pd.DataFrame) -> InsightResult:
    """Do vendors that eventually churn underperform the survivors?"""
    churned_ids = outcomes.groupby("vendor_id")["churned"].max()
    avg_perf = performance.groupby("vendor_id")["overall_score"].mean()

    joined = pd.concat([churned_ids.rename("churned"), avg_perf.rename("score")], axis=1).dropna()
    churned = joined.loc[joined["churned"] == 1, "score"].to_numpy()
    survived = joined.loc[joined["churned"] == 0, "score"].to_numpy()

    t, p = stats.ttest_ind(churned, survived, equal_var=False)
    d = _cohens_d(churned, survived)
    return InsightResult(
        question="Do churned vendors underperform before leaving?",
        test_name="Welch's t-test",
        statistic=round(float(t), 3),
        p_value=round(float(p), 4),
        effect_size=round(d, 3),
        effect_name="Cohen's d",
        significant=p < ALPHA,
        interpretation=(
            f"Churned vendors averaged {churned.mean():.1f} vs {survived.mean():.1f} "
            f"for retained vendors (d={d:.2f}, "
            f"{'significant' if p < ALPHA else 'not significant'})."
        ),
    )


def category_performance_anova(performance: pd.DataFrame) -> InsightResult:
    """Does average performance differ across vendor categories?"""
    groups = [g["overall_score"].dropna().to_numpy() for _, g in performance.groupby("category")]
    f, p = stats.f_oneway(*groups)

    grand = performance["overall_score"].dropna()
    ss_between = sum(len(g) * (g.mean() - grand.mean()) ** 2 for g in groups)
    ss_total = float(((grand - grand.mean()) ** 2).sum())
    eta_sq = ss_between / ss_total if ss_total else 0.0

    best = performance.groupby("category")["overall_score"].mean().idxmax()
    worst = performance.groupby("category")["overall_score"].mean().idxmin()
    return InsightResult(
        question="Does performance differ across vendor categories?",
        test_name="One-way ANOVA",
        statistic=round(float(f), 3),
        p_value=round(float(p), 4),
        effect_size=round(float(eta_sq), 3),
        effect_name="Eta-squared",
        significant=p < ALPHA,
        interpretation=(
            f"Strongest category: {best}; weakest: {worst} "
            f"(η²={eta_sq:.3f} of variance explained by category)."
        ),
    )


def sla_breach_vs_relationship(outcomes: pd.DataFrame) -> InsightResult:
    """Are SLA breaches associated with weaker relationship health?"""
    table = pd.crosstab(outcomes["sla_breach_flag"], outcomes["relationship_health"])
    chi2, p, _, _ = stats.chi2_contingency(table)
    v = _cramers_v(table, chi2)
    return InsightResult(
        question="Are SLA breaches associated with weaker relationships?",
        test_name="Chi-squared test of independence",
        statistic=round(float(chi2), 3),
        p_value=round(float(p), 4),
        effect_size=round(v, 3),
        effect_name="Cramér's V",
        significant=p < ALPHA,
        interpretation=(
            f"SLA breaches and relationship health are "
            f"{'dependent' if p < ALPHA else 'independent'} (V={v:.2f})."
        ),
    )


def spend_roi_correlation(financial: pd.DataFrame) -> InsightResult:
    """Is higher spend associated with better ROI scores?"""
    df = financial[["total_spend", "roi_score"]].dropna()
    r, p = stats.pearsonr(df["total_spend"], df["roi_score"])
    return InsightResult(
        question="Does higher spend buy better ROI?",
        test_name="Pearson correlation",
        statistic=round(float(r), 3),
        p_value=round(float(p), 4),
        effect_size=round(float(r), 3),
        effect_name="Pearson r",
        significant=p < ALPHA,
        interpretation=(
            f"Spend and ROI are {'positively' if r > 0 else 'negatively'} correlated "
            f"(r={r:.2f}) — "
            + (
                "spending more is associated with better returns."
                if r > 0.1
                else "no meaningful linear relationship; spend alone doesn't buy ROI."
                if abs(r) <= 0.1
                else "bigger contracts are seeing worse returns — review pricing."
            )
        ),
    )


def run_all_insights(
    outcomes: pd.DataFrame, performance: pd.DataFrame, financial: pd.DataFrame
) -> list[InsightResult]:
    """Run the full battery; skip gracefully on missing data."""
    insights: list[InsightResult] = []
    for fn, args in [
        (escalation_vs_renewal, (outcomes,)),
        (performance_vs_churn, (outcomes, performance)),
        (category_performance_anova, (performance,)),
        (sla_breach_vs_relationship, (outcomes,)),
        (spend_roi_correlation, (financial,)),
    ]:
        try:
            insights.append(fn(*args))
        except Exception:
            continue
    return insights
