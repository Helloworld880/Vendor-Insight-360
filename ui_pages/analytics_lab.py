"""Analytics Lab — cohorts, funnel, hypothesis tests and segmentation.

Every chart carries a one-line takeaway and the statistical evidence
behind it. Null results are shown too: knowing spend does NOT buy ROI
is as actionable as any significant finding.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core_modules.cohort_analysis import (
    cohort_retention_matrix,
    quarterly_retention,
    renewal_funnel,
)
from core_modules.stats_tests import run_all_insights
from core_modules.vendor_clustering import segment_vendors

RISK_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}


@st.cache_data(ttl=3600, show_spinner=False)
def _load_frames(_db):
    return (
        _db.get_vendor_outcomes(),
        _db.get_performance_data(),
        _db.get_financial_data(),
        _db.get_vendors(),
    )


def render_analytics_lab(db) -> None:
    st.markdown(
        '<div class="main-header">🔬 Analytics Lab</div>', unsafe_allow_html=True
    )
    st.caption(
        "Cohort retention, funnel conversion, statistical hypothesis tests and "
        "vendor segmentation — each with the evidence behind the headline."
    )

    outcomes, performance, financial, vendors = _load_frames(db)
    if outcomes.empty or performance.empty:
        st.warning("Outcome / performance datasets not found in `Data layer/`.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Cohorts & Retention", "🔻 Renewal Funnel", "🧪 Statistical Insights", "🎯 Segmentation"]
    )

    # ── Cohorts & retention ──────────────────────────────────────────────
    with tab1:
        st.subheader("Survival by initial-performance cohort")
        st.caption(
            "Vendors are cohorted by the quartile of their first-quarter "
            "performance score. Question: do vendors that start strong stay longer? "
            "(Join-date cohorts are degenerate here — all contracts share a start date.)"
        )
        matrix = cohort_retention_matrix(outcomes, performance)
        fig = px.imshow(
            matrix,
            text_auto=".1f",
            color_continuous_scale="RdYlGn",
            zmin=80,
            zmax=100,
            labels={"x": "Quarter", "y": "Cohort", "color": "% surviving"},
            title="Vendor survival rate (%) by cohort and quarter",
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)

        weakest = matrix.iloc[:, -1].idxmin()
        strongest = matrix.iloc[:, -1].idxmax()
        st.info(
            f"**Takeaway:** by the latest quarter, the *{strongest}* cohort retains "
            f"{matrix.iloc[:, -1].max():.1f}% of vendors vs {matrix.iloc[:, -1].min():.1f}% "
            f"for *{weakest}* — early performance is a leading indicator of retention."
        )

        st.subheader("Quarterly churn & cumulative survival")
        ret = quarterly_retention(outcomes)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=ret["quarter"], y=ret["churn_rate_pct"], name="Churn rate (%)",
                   marker_color="#ef4444")
        )
        fig.add_trace(
            go.Scatter(x=ret["quarter"], y=ret["cumulative_survival_pct"],
                       name="Cumulative survival (%)", yaxis="y2",
                       mode="lines+markers", line={"color": "#3b82f6"})
        )
        fig.update_layout(
            yaxis={"title": "Quarterly churn rate (%)"},
            yaxis2={"title": "Cumulative survival (%)", "overlaying": "y", "side": "right"},
            title="Portfolio attrition over time",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Funnel ───────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Vendor lifecycle funnel")
        funnel = renewal_funnel(vendors, outcomes)
        fig = go.Figure(
            go.Funnel(
                y=funnel["stage"],
                x=funnel["vendors"],
                textinfo="value+percent initial",
                marker={"color": ["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#22c55e"]},
            )
        )
        fig.update_layout(title="From onboarding to loyal partnership")
        st.plotly_chart(fig, use_container_width=True)

        biggest_drop = funnel.loc[funnel["drop_off"].idxmax()]
        st.info(
            f"**Takeaway:** the largest drop-off ({int(biggest_drop['drop_off'])} vendors) "
            f"happens entering **{biggest_drop['stage']}** — that's where retention "
            "effort buys the most."
        )
        st.dataframe(funnel, use_container_width=True, hide_index=True)

    # ── Statistical insights ─────────────────────────────────────────────
    with tab3:
        st.subheader("Hypothesis tests with effect sizes")
        st.caption(
            "p-values alone overstate findings on large samples — every test below "
            "reports an effect size, and null results are kept on the board."
        )
        insights = run_all_insights(outcomes, performance, financial)
        for ins in insights:
            icon = "✅" if ins.significant else "⚪️"
            with st.expander(f"{icon} {ins.question}", expanded=ins.significant):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Test", ins.test_name.split(" ")[0])
                c2.metric("Statistic", f"{ins.statistic:.3f}")
                c3.metric("p-value", f"{ins.p_value:.4f}")
                c4.metric(ins.effect_name, f"{ins.effect_size:.3f}")
                st.write(ins.interpretation)

        sig = sum(1 for i in insights if i.significant)
        st.info(
            f"**Takeaway:** {sig} of {len(insights)} hypotheses are statistically "
            "significant. Escalations are the strongest renewal signal — they belong "
            "in any churn playbook."
        )

    # ── Segmentation ─────────────────────────────────────────────────────
    with tab4:
        st.subheader("K-Means vendor segmentation")
        st.caption(
            "Features are standardised; k is chosen by silhouette score across "
            "k=2…6 rather than hardcoded. Segments are profiled into business personas."
        )
        vwp = db.get_vendors_with_performance()
        if vwp.empty:
            st.warning("No vendor performance data available.")
            return

        result = segment_vendors(vwp, financial)
        c1, c2 = st.columns(2)
        c1.metric("Chosen k (by silhouette)", result.k)
        c2.metric("Silhouette score", f"{result.silhouette:.3f}")

        fig = px.scatter(
            result.segments,
            x="avg_performance",
            y="total_spend",
            color="segment",
            size="avg_defect_rate",
            hover_name="name",
            title="Vendor segments: performance vs total spend (size = defect rate)",
            labels={"avg_performance": "Avg performance score", "total_spend": "Total spend ($)"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Segment profiles")
        st.dataframe(result.profile, use_container_width=True, hide_index=True)

        watch = result.segments[result.segments["segment"].str.contains("Watch", na=False)]
        if not watch.empty:
            exposure = watch["total_spend"].sum()
            st.warning(
                f"**Action:** {len(watch)} vendors sit in the Watch List segment "
                f"(high spend, low performance) — ${exposure:,.0f} of spend deserves "
                "a renegotiation or remediation conversation."
            )
