import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="ML Vendor Optimization Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass

warnings.filterwarnings("ignore")


# ── Local modules ────────────────────────────────────────────────────────────
from core_modules.auth import Authentication
from core_modules.database import DatabaseManager
from core_modules.analytics import AnalyticsEngine
from core_modules.config import Config
from ui_pages.ai_page import render_ai_workspace as render_ai_workspace_page
from ui_pages.analytics_lab import render_analytics_lab as render_analytics_lab_page
from ui_pages.reports_page import render_reports as render_reports_page
from ui_pages.risk_page import render_risk_management as render_risk_management_page
from ui_pages.settings_page import render_settings as render_settings_page

# ML Engine (lazy-loaded)
_ML_AVAILABLE = False
MLEngine = None
try:
    from enhancements.ml_engine import MLEngine as _MLEngine
    MLEngine = _MLEngine
    _ML_AVAILABLE = True
except ImportError as e:
    st.warning(f"ML engine not available: {e}. Install scikit-learn.")

# Report generator (lazy-loaded)
_REPORT_AVAILABLE = False
ReportGenerator = None
try:
    from enhancements.report_generator import ReportGenerator as _RG
    ReportGenerator = _RG
    _REPORT_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
def inject_styles():
    st.markdown("""
    <style>

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #1F3C88 !important;
    }

    /* Login input text color fix */
    input[type="text"], input[type="password"] {
        background-color: white !important;
        color: black !important;
        border-radius: 8px !important;
        border: 1px solid #c7d2fe !important;
        padding: 8px !important;
    }

    /* Fix label color */
    label {
        color: #dbeafe !important;
        font-weight: 500;
    }

    /* Login button */
    button[kind="primary"] {
        background: linear-gradient(135deg,#2563eb,#4f46e5) !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
    }

    .hero-panel {
        background: linear-gradient(135deg, #eef4ff 0%, #f9fbff 60%, #ffffff 100%);
        border: 1px solid #d6e4ff;
        border-radius: 18px;
        padding: 20px 22px;
        margin-bottom: 18px;
        box-shadow: 0 12px 30px rgba(31, 60, 136, 0.08);
    }

    .hero-kicker {
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #3b82f6;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .hero-title {
        font-size: 1.8rem;
        color: #12264d;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .hero-copy {
        color: #4b5563;
        line-height: 1.55;
        margin-bottom: 0;
    }

    .priority-card {
        border: 1px solid #e5e7eb;
        border-left: 6px solid #1f3c88;
        background: #ffffff;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }

    .priority-high { border-left-color: #ef4444; }
    .priority-medium { border-left-color: #f59e0b; }
    .priority-low { border-left-color: #22c55e; }

    .pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 8px;
    }

    .pill-high { background: #fee2e2; color: #b91c1c; }
    .pill-medium { background: #fef3c7; color: #92400e; }
    .pill-low { background: #dcfce7; color: #166534; }
    .pill-neutral { background: #e5e7eb; color: #374151; }

    .insight-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }

    .insight-title {
        font-weight: 800;
        color: #12264d;
        margin-bottom: 6px;
    }

    
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def to_df(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if obj is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(list(obj))
    except Exception:
        return pd.DataFrame()


def fmt_currency(val):
    if val >= 1_000_000:
        return f"${val/1_000_000:.1f}M"
    if val >= 1_000:
        return f"${val/1_000:.0f}K"
    return f"${val:,.0f}"


def risk_color(level: str) -> str:
    return {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}.get(level, "#6b7280")


def format_pct(val):
    try:
        return f"{float(val):.1f}%"
    except Exception:
        return "—"


def status_tone(level: str) -> str:
    return {
        "High": "high",
        "Medium": "medium",
        "Low": "low",
        "Compliant": "low",
        "Under Review": "medium",
        "Non-Compliant": "high",
    }.get(str(level), "neutral")


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD CLASS
# ─────────────────────────────────────────────────────────────────────────────
class VendorDashboard:
    def __init__(self):
        self.config = Config()
        self.db = DatabaseManager()
        self.auth = Authentication(self.db)
        self.analytics = AnalyticsEngine(self.db)
        self._ml: object = None
        self._report_gen: object = None
        self._init_session()

    # ── Lazy ML ──────────────────────────────────────────────────────────────
    @property
    def ml(self):
        if self._ml is None and _ML_AVAILABLE and MLEngine:
            with st.spinner("🤖 Loading ML models…"):
                self._ml = MLEngine(self.db)
        return self._ml

    @property
    def report_gen(self):
        if self._report_gen is None and _REPORT_AVAILABLE and ReportGenerator:
            self._report_gen = ReportGenerator(self.db)
        return self._report_gen

    def _churn_predictor(self):
        """Train (once per session) the supervised churn model on labelled outcomes."""
        from core_modules.churn_model import ChurnPredictor

        @st.cache_resource(show_spinner=False)
        def _train():
            outcomes = self.db.get_vendor_outcomes()
            performance = self.db.get_performance_data()
            financial = self.db.get_financial_data()
            risk = self.db.get_risk_history()
            if outcomes.empty or performance.empty:
                return None, None
            predictor = ChurnPredictor()
            metrics = predictor.train(outcomes, performance, financial, risk)
            return predictor, metrics

        return _train()

    # ── Session ──────────────────────────────────────────────────────────────
    def _init_session(self):
        defaults = {
            "user": None,
            "selected_nav": "🏠 Overview",
            "perf_threshold": 70,
            "filters": {},
            "last_activity_at": datetime.now(),
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    def _dataset_inventory(self):
        data_dir = os.path.join(os.getcwd(), "Data layer")
        if not os.path.exists(data_dir):
            return pd.DataFrame()

        rows = []
        for name in sorted(os.listdir(data_dir)):
            path = os.path.join(data_dir, name)
            if not os.path.isfile(path):
                continue
            rows.append(
                {
                    "dataset": name,
                    "size_kb": round(os.path.getsize(path) / 1024, 1),
                    "updated": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M"),
                }
            )
        return pd.DataFrame(rows)

    @dataclass
    class DataHealth:
        label: str
        rows: int
        updated: str
        source: str

    def _file_updated_str(self, rel_path: str) -> str:
        try:
            if os.path.exists(rel_path):
                return datetime.fromtimestamp(os.path.getmtime(rel_path)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
        return "—"

    def _data_health(self) -> list["VendorDashboard.DataHealth"]:
        perf_df, fin_df, perf_history, compliance, risk = self._get_ai_dataframes()
        vendors = to_df(self.db.get_vendors())

        db_path = getattr(self.db, "db_path", "Data layer/vendors.db")
        inventory = self._dataset_inventory()
        csv_set = set(inventory["dataset"].tolist()) if not inventory.empty and "dataset" in inventory.columns else set()

        def _source_for(primary_csv: str) -> str:
            return "csv" if primary_csv in csv_set else "db"

        return [
            self.DataHealth("Vendors", len(vendors), self._file_updated_str(db_path), _source_for("vendors.csv")),
            self.DataHealth("Performance (latest)", len(perf_df), self._file_updated_str("Data layer/performance.csv"), _source_for("performance.csv")),
            self.DataHealth("Performance history", len(perf_history), self._file_updated_str("Data layer/performance.csv"), _source_for("performance.csv")),
            self.DataHealth("Financial", len(fin_df), self._file_updated_str("Data layer/financial_metrics.csv"), _source_for("financial_metrics.csv")),
            self.DataHealth("Compliance", len(compliance), self._file_updated_str("Data layer/compliance_history.csv"), _source_for("compliance_history.csv")),
            self.DataHealth("Risk", len(risk), self._file_updated_str("Data layer/risk_history.csv"), _source_for("risk_history.csv")),
        ]

    def _render_data_health_panel(self):
        with st.expander("🩺 Data Health", expanded=False):
            rows = self._data_health()
            c1, c2, c3 = st.columns([1.2, 0.7, 1.1])
            c1.markdown("**Dataset**")
            c2.markdown("**Rows**")
            c3.markdown("**Updated / Source**")

            for item in rows:
                a, b, c = st.columns([1.2, 0.7, 1.1])
                a.write(item.label)
                b.write(item.rows)
                c.write(f"{item.updated}  ·  {item.source.upper()}")

            if any(i.rows == 0 for i in rows if i.label in {"Performance (latest)", "Risk", "Compliance"}):
                st.caption("Tip: if pages look blank, it usually means these datasets have 0 rows.")

    def _save_uploaded_dataset(self, uploaded_file, target_name: str):
        data_dir = os.path.join(os.getcwd(), "Data layer")
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, target_name)
        with open(path, "wb") as handle:
            handle.write(uploaded_file.getbuffer())
        return path

    def _get_risk_review_frame(self):
        perf_df, fin_df, _, compliance, risk = self._get_ai_dataframes()
        review = perf_df.copy()
        if review.empty:
            return review

        if "overall_risk" not in review.columns:
            review["overall_risk"] = np.nan
        if "compliance_score" not in review.columns:
            review["compliance_score"] = np.nan

        review["priority_score"] = (
            review["overall_risk"].fillna(0) * 0.45
            + (100 - review["performance_score"].fillna(0)) * 0.30
            + (100 - review["compliance_score"].fillna(0)) * 0.25
        ).round(1)
        review = review.sort_values("priority_score", ascending=False)

        if not fin_df.empty:
            fin_cols = [c for c in ["vendor_name", "cost_variance", "payment_days", "overdue_invoices", "roi_score"] if c in fin_df.columns]
            review = review.merge(
                fin_df[fin_cols],
                on="vendor_name",
                how="left",
            )

        if not compliance.empty:
            review = review.merge(
                compliance[["vendor_name", "compliance_status", "next_audit_date"]].drop_duplicates("vendor_name", keep="last"),
                on="vendor_name",
                how="left",
            )

        if not risk.empty:
            extra_cols = [c for c in ["vendor_name", "financial_risk", "operational_risk", "compliance_risk", "mitigation_status", "risk_level", "overall_risk"] if c in risk.columns]
            if extra_cols:
                review = review.merge(risk[extra_cols].drop_duplicates("vendor_name", keep="last"), on="vendor_name", how="left")

        if "overall_risk_x" in review.columns or "overall_risk_y" in review.columns:
            review["overall_risk"] = review.get("overall_risk_x").combine_first(review.get("overall_risk_y"))
            review = review.drop(columns=[c for c in ["overall_risk_x", "overall_risk_y"] if c in review.columns])

        if "risk_level_x" in review.columns or "risk_level_y" in review.columns:
            review["risk_level"] = review.get("risk_level_x").combine_first(review.get("risk_level_y"))
            review = review.drop(columns=[c for c in ["risk_level_x", "risk_level_y"] if c in review.columns])

        if "overall_risk" not in review.columns:
            review["overall_risk"] = np.nan
        if review["overall_risk"].isna().all() and all(c in review.columns for c in ["financial_risk", "operational_risk", "compliance_risk"]):
            review["overall_risk"] = (
                review["financial_risk"].fillna(0) * 0.4
                + review["operational_risk"].fillna(0) * 0.35
                + review["compliance_risk"].fillna(0) * 0.25
            ).round(1)

        if "risk_level" not in review.columns:
            review["risk_level"] = "Low"
        missing_level = review["risk_level"].isna() | (review["risk_level"].astype(str).str.strip() == "")
        review.loc[missing_level & (review["overall_risk"] >= 60), "risk_level"] = "High"
        review.loc[missing_level & (review["overall_risk"].between(35, 59.999, inclusive="both")), "risk_level"] = "Medium"
        review.loc[missing_level & (review["overall_risk"] < 35), "risk_level"] = "Low"

        return review

    def _render_priority_card(self, row):
        tone = status_tone(row.get("risk_level", "neutral"))
        pill_tone = tone if tone in {"high", "medium", "low"} else "neutral"
        status = row.get("compliance_status", "No status")
        st.markdown(
            f"""
            <div class="priority-card priority-{tone}">
                <div>
                    <span class="pill pill-{pill_tone}">{row.get('risk_level', 'Unknown')} risk</span>
                    <span class="pill pill-{status_tone(status)}">{status}</span>
                </div>
                <div style="font-size:1.05rem;font-weight:800;color:#12264d;margin:8px 0 4px 0;">{row.get('vendor_name', 'Vendor')}</div>
                <div style="color:#4b5563;font-size:0.92rem;">
                    Priority score: <strong>{row.get('priority_score', '—')}</strong> |
                    Performance: <strong>{format_pct(row.get('performance_score'))}</strong> |
                    Compliance: <strong>{format_pct(row.get('compliance_score'))}</strong> |
                    Overall risk: <strong>{format_pct(row.get('overall_risk'))}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _risk_action_recommendations(self, row, outcome=None):
        actions = []
        if float(row.get("overall_risk", 0) or 0) >= 60:
            actions.append("Escalate this vendor in the weekly operating review and assign a named owner.")
        if float(row.get("financial_risk", 0) or 0) >= 55:
            actions.append("Validate exposure, contract concentration, and any pending payment disputes.")
        if float(row.get("operational_risk", 0) or 0) >= 50:
            actions.append("Run a service delivery check on SLA misses, delays, and open operational blockers.")
        if float(row.get("compliance_risk", 0) or 0) >= 45:
            actions.append("Pull the latest audit evidence and confirm a date for the next compliance review.")
        if str(row.get("mitigation_status", "")).lower() in {"monitoring", "in progress"}:
            actions.append("Update mitigation progress this week and capture the next checkpoint in writing.")
        if outcome is not None:
            if int(outcome.get("escalation_flag", 0) or 0) == 1:
                actions.append("Previous period had an escalation, so align commercial and delivery owners before renewal discussions.")
            if int(outcome.get("sla_breach_flag", 0) or 0) == 1:
                actions.append("SLA breach history exists, so agree on one measurable recovery commitment with the vendor.")
        if not actions:
            actions.append("Keep this vendor on a light monitoring cadence and review again after the next assessment.")
        return actions[:4]

    def _risk_leadership_note(self, row, trend_delta, outcome=None):
        vendor = row.get("vendor_name", "This vendor")
        level = str(row.get("risk_level", "Unknown")).lower()
        overall = format_pct(row.get("overall_risk"))
        mitigation = row.get("mitigation_status", "not recorded")
        trend_text = "stable"
        if pd.notna(trend_delta):
            if trend_delta >= 5:
                trend_text = f"worsening by {trend_delta:.1f} points versus the previous review"
            elif trend_delta <= -5:
                trend_text = f"improving by {abs(trend_delta):.1f} points versus the previous review"
        outcome_text = ""
        if outcome is not None and not outcome.empty:
            rec = outcome.iloc[0]
            outcome_text = (
                f" Relationship health is {str(rec.get('relationship_health', 'unknown')).lower()} and the latest quarter "
                f"shows {int(rec.get('incident_count', 0) or 0)} incident(s)."
            )
        return (
            f"{vendor} is currently rated {level} risk with an overall score of {overall}. "
            f"The profile is {trend_text}, and mitigation is marked as {mitigation}.{outcome_text}"
        )

    def _get_ai_dataframes(self):
        perf = to_df(self.db.get_vendors_with_performance())
        perf_history = to_df(self.db.get_performance_data())
        financial = to_df(self.db.get_financial_data())
        compliance = to_df(self.db.get_compliance_data())
        risk = to_df(self.db.get_risk_data())

        perf_df = perf.copy()
        if not perf_df.empty:
            perf_df = perf_df.rename(
                columns={
                    "name": "vendor_name",
                    "avg_performance": "performance_score",
                    "avg_on_time": "on_time_delivery",
                    "avg_quality": "quality_score",
                }
            )

            if not compliance.empty:
                comp_summary = (
                    compliance.sort_values("audit_date")
                    .drop_duplicates("vendor_name", keep="last")[["vendor_name", "audit_score", "compliance_status"]]
                    .rename(columns={"audit_score": "compliance_score"})
                )
                perf_df = perf_df.merge(comp_summary, on="vendor_name", how="left")

            if not risk.empty:
                risk_summary = (
                    risk.sort_values("assessment_date")
                    .drop_duplicates("vendor_name", keep="last")[["vendor_name", "overall_risk", "risk_level"]]
                )
                perf_df = perf_df.merge(risk_summary, on="vendor_name", how="left")

            if "compliance_score" not in perf_df.columns:
                perf_df["compliance_score"] = perf_df.get("performance_score", 0)

            perf_df = perf_df[
                [
                    col
                    for col in [
                        "vendor_name",
                        "category",
                        "status",
                        "risk_level",
                        "contract_value",
                        "performance_score",
                        "on_time_delivery",
                        "quality_score",
                        "compliance_score",
                        "overall_risk",
                        "country",
                    ]
                    if col in perf_df.columns
                ]
            ]

        fin_df = financial.copy()
        if not fin_df.empty:
            agg_map = {"total_spend": "sum", "cost_savings": "sum"}
            if "invoice_count" in fin_df.columns:
                agg_map["invoice_count"] = "sum"
            if "payment_days" in fin_df.columns:
                agg_map["payment_days"] = "mean"
            if "overdue_invoices" in fin_df.columns:
                agg_map["overdue_invoices"] = "sum"
            if "roi_score" in fin_df.columns:
                agg_map["roi_score"] = "mean"
            fin_df = fin_df.groupby("vendor_name", as_index=False).agg(agg_map)
            contract_lookup = perf[["name", "contract_value"]].rename(columns={"name": "vendor_name"}) if not perf.empty else pd.DataFrame()
            if not contract_lookup.empty:
                fin_df = fin_df.merge(contract_lookup, on="vendor_name", how="left")
            fin_df["planned_cost"] = fin_df["total_spend"] - fin_df["cost_savings"]
            fin_df["actual_cost"] = fin_df["total_spend"]
            fin_df["cost_variance"] = fin_df["actual_cost"] - fin_df["planned_cost"]
            fin_df = fin_df[
                [
                    col
                    for col in [
                        "vendor_name",
                        "contract_value",
                        "planned_cost",
                        "actual_cost",
                        "cost_variance",
                        "invoice_count",
                        "payment_days",
                        "overdue_invoices",
                        "roi_score",
                    ]
                    if col in fin_df.columns
                ]
            ]

        return perf_df, fin_df, perf_history, compliance, risk

    def render_ai_workspace(self):
        render_ai_workspace_page(self)

    def render_analytics_lab(self):
        render_analytics_lab_page(self.db)

    # ─────────────────────────────────────────────────────────────────────────
    # SIDEBAR
    # ─────────────────────────────────────────────────────────────────────────
    def render_sidebar(self):
        with st.sidebar:
            st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
            st.title("Vendor Insight360")
            st.caption(f"v{self.config.APP_VERSION}")
            st.divider()

            # ── Login / logout ──
            if st.session_state.user is None:
                st.subheader("🔐 Login")
                with st.form("login_form", clear_on_submit=False):
                    uname = st.text_input("Username", placeholder=self.config.DEMO_ADMIN_USERNAME)
                    pwd = st.text_input("Password", type="password", placeholder=self.config.DEMO_ADMIN_PASSWORD)
                    if st.form_submit_button("Login", use_container_width=True):
                        user = self.auth.authenticate(uname, pwd)
                        if user:
                            st.session_state.user = user
                            st.session_state.last_activity_at = datetime.now()
                            st.success(f"Welcome, {user['name']}!")
                            st.rerun()
                        else:
                            st.error("Invalid credentials")
                st.info(f"Demo: {self.config.DEMO_ADMIN_USERNAME} / {self.config.DEMO_ADMIN_PASSWORD}")
                return

            user = st.session_state.user
            st.success(f"👋 {user['name']}")
            st.caption(f"Role: **{user['role'].upper()}**")
            if st.button("Logout", use_container_width=True):
                st.session_state.user = None
                st.session_state.last_activity_at = datetime.now()
                st.rerun()

            st.divider()
            self._render_data_health_panel()
            st.divider()
            st.subheader("📌 Navigation")
            nav_options = [
                "🏠 Overview",
                "📊 Vendor Performance",
                "💰 Financial Analytics",
                "⚠️ Risk Management",
                "📋 Compliance",
                "🧠 AI Insights",
                "🤖 ML Predictions",
                "🔬 Analytics Lab",
                "📄 Reports",
                "🏢 Vendor Portal",
                "⚙️ Settings",
            ]
            st.session_state.selected_nav = st.selectbox(
                "Go to", nav_options,
                index=nav_options.index(st.session_state.selected_nav)
                if st.session_state.selected_nav in nav_options else 0,
            )

            st.divider()
            st.subheader("🔧 Filters")
            st.session_state.perf_threshold = st.slider(
                "Performance Threshold", 0, 100, st.session_state.perf_threshold)
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. OVERVIEW
    # ─────────────────────────────────────────────────────────────────────────
    def render_overview(self):
        st.markdown('<div class="main-header">🏠 Vendor Performance Overview</div>',
                unsafe_allow_html=True)

        vendors = to_df(self.db.get_vendors())
        vendors_perf = to_df(self.db.get_vendors_with_performance())
        financial = to_df(self.db.get_financial_data())

        # ── KPI Calculations ─────────────────────────────
        total_vendors = len(vendors)

        active_vendors = 0
        if not vendors.empty and "status" in vendors.columns:
            active_vendors = (vendors["status"].str.lower() == "active").sum()

        avg_performance = 0
        if not vendors_perf.empty and "avg_performance" in vendors_perf.columns:
            avg_performance = vendors_perf["avg_performance"].mean()

        high_risk = 0
        if not vendors.empty and "risk_level" in vendors.columns:
            high_risk = (vendors["risk_level"].str.lower() == "high").sum()

        total_contract_value = 0
        if not vendors.empty and "contract_value" in vendors.columns:
            total_contract_value = vendors["contract_value"].sum()

        total_cost_savings = 0
        if not financial.empty and "cost_savings" in financial.columns:
            total_cost_savings = financial["cost_savings"].sum()

        # ── KPI Display ─────────────────────────────────
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        c1.metric("Total Vendors", total_vendors)
        c2.metric("Active Vendors", active_vendors)
        c3.metric("Avg Performance", f"{avg_performance:.1f}%")
        c4.metric("High Risk Vendors", high_risk)
        c5.metric("Total Contract Value", fmt_currency(total_contract_value))
        c6.metric("Total Cost Savings", fmt_currency(total_cost_savings))

        st.divider()

        # ── Charts ──────────────────────────────────────
        vendors = vendors_perf

        col1, col2 = st.columns(2)
    

        with col1:
            st.subheader("Top Vendors by Contract Value")
            if not vendors.empty:
                top = vendors.nlargest(10, "contract_value")
                fig = px.bar(
                    top,
                    x="contract_value",
                    y="name",
                    orientation="h",
                    color="risk_level",
                    color_discrete_map={
                        "High": "#ef4444",
                        "Medium": "#f59e0b",
                        "Low": "#22c55e",
                    },
                    labels={"contract_value": "Contract Value ($)", "name": ""},
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No vendor data available.")

        with col2:
            st.subheader("Performance Trend")

            trend = to_df(self.db.get_performance_trends())

            if not trend.empty:
                trend["metric_date"] = pd.to_datetime(trend["metric_date"])

                fig = px.line(
                    trend,
                    x="metric_date",
                    y="avg_score",
                    markers=True,
                    labels={
                        "metric_date": "Date",
                        "avg_score": "Average Score",
                    },
                )

                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No performance trend data.")

        # ─────────────────────────────────────────────────────────────────────────
        # 2. VENDOR PERFORMANCE
        # ─────────────────────────────────────────────────────────────────────────
    def render_vendor_performance(self):
            st.markdown('<div class="main-header">📊 Vendor Performance Analysis</div>',
                        unsafe_allow_html=True)

            vendors = to_df(self.db.get_vendors_with_performance())
            if vendors.empty:
                st.warning("No performance data found.")
                return

            # Filter bar
            with st.expander("🔎 Filters", expanded=True):
                fc1, fc2, fc3 = st.columns(3)
                cats = ["All"] + sorted(vendors["category"].dropna().unique().tolist())
                risks = ["All", "High", "Medium", "Low"]
                sel_cat = fc1.selectbox("Category", cats)
                sel_risk = fc2.selectbox("Risk Level", risks)
                min_cv, max_cv = int(vendors["contract_value"].min()), int(vendors["contract_value"].max())
                cv_range = fc3.slider("Contract Value ($)", min_cv, max_cv, (min_cv, max_cv), step=5000)

            filt = vendors.copy()
            if sel_cat != "All":
                filt = filt[filt["category"] == sel_cat]
            if sel_risk != "All":
                filt = filt[filt["risk_level"] == sel_risk]
            filt = filt[(filt["contract_value"] >= cv_range[0]) & (filt["contract_value"] <= cv_range[1])]

            st.success(f"Showing **{len(filt)}** vendors")

            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Avg Performance", f"{filt['avg_performance'].mean():.1f}%" if "avg_performance" in filt else "—")
            k2.metric("Avg On-Time %", f"{filt['avg_on_time'].mean():.1f}%" if "avg_on_time" in filt else "—")
            k3.metric("Avg Defect Rate", f"{filt['avg_defect_rate'].mean():.2f}%" if "avg_defect_rate" in filt else "—")
            k4.metric("Avg Quality Score", f"{filt['avg_quality'].mean():.1f}%" if "avg_quality" in filt else "—")

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(filt, x="avg_on_time", y="avg_quality",
                                color="risk_level", size="contract_value",
                                hover_name="name", hover_data=["category"],
                                color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"},
                                title="On-Time vs Quality Score (bubble = contract value)",
                                labels={"avg_on_time": "On-Time Delivery (%)", "avg_quality": "Quality Score (%)"})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.box(filt, x="category", y="avg_performance",
                            color="category", title="Performance Distribution by Category",
                            labels={"avg_performance": "Performance Score (%)"})
                fig.update_layout(xaxis_tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)

            # Heatmap
            st.subheader("📊 Vendor Metrics Heatmap")
            heat_cols = [c for c in ["avg_performance", "avg_on_time", "avg_quality", "avg_defect_rate", "contract_value"]
                        if c in filt.columns]
            if len(heat_cols) >= 2 and not filt.empty:
                heat_data = filt[["name"] + heat_cols].set_index("name")[heat_cols]
                heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min() + 1e-9)
                fig = go.Figure(data=go.Heatmap(
                    z=heat_norm.values, x=heat_cols, y=heat_norm.index.tolist(),
                    colorscale="RdYlGn", text=heat_data.values.round(1),
                    texttemplate="%{text}", textfont={"size": 9},
                ))
                fig.update_layout(height=max(300, len(filt) * 22), yaxis_autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)

            # Vendor comparison
            st.subheader("🔄 Head-to-Head Vendor Comparison")
            v_options = filt["name"].dropna().unique().tolist()
            vc1, vc2 = st.columns(2)
            sel_v1 = vc1.selectbox("Vendor A", v_options, key="va")
            sel_v2 = vc2.selectbox("Vendor B", [v for v in v_options if v != sel_v1], key="vb")
            if sel_v1 and sel_v2:
                r1 = filt[filt["name"] == sel_v1].iloc[0]
                r2 = filt[filt["name"] == sel_v2].iloc[0]
                comp_metrics = ["avg_performance", "avg_on_time", "avg_quality"]
                fig = go.Figure()
                fig.add_trace(go.Bar(name=sel_v1, x=comp_metrics,
                                    y=[r1.get(m, 0) for m in comp_metrics], marker_color="#1f3c88"))
                fig.add_trace(go.Bar(name=sel_v2, x=comp_metrics,
                                    y=[r2.get(m, 0) for m in comp_metrics], marker_color="#df3e16"))
                fig.update_layout(barmode="group", title="Performance Comparison")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 Vendor Table")
            st.dataframe(filt.round(2), use_container_width=True)
            st.download_button("📥 Download CSV", filt.to_csv(index=False).encode(),
                            "vendor_performance.csv", "text/csv")

        # ─────────────────────────────────────────────────────────────────────────
        # 3. FINANCIAL ANALYTICS
        # ─────────────────────────────────────────────────────────────────────────
    def render_financial_analytics(self):
            st.markdown('<div class="main-header">💰 Financial Analytics</div>',
                        unsafe_allow_html=True)

            fin_summary = to_df(self.db.get_financial_summary())
            fin_detail = to_df(self.db.get_financial_data())

            if fin_summary.empty:
                st.warning("No financial data found.")
                return

            # KPIs
            total_spend = fin_summary["total_spend"].sum()
            total_savings = fin_summary["cost_savings"].sum()
            savings_rate = (total_savings / total_spend * 100) if total_spend > 0 else 0
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Spend",         fmt_currency(total_spend))
            k2.metric("Total Cost Savings",   fmt_currency(total_savings))
            k3.metric("Overall Savings Rate", f"{savings_rate:.1f}%")
            k4.metric("Categories",           len(fin_summary))

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                fin_summary["savings_rate"] = (fin_summary["cost_savings"] / fin_summary["total_spend"] * 100).round(1)
                fig = px.bar(fin_summary, x="category", y="total_spend",
                            color="savings_rate", color_continuous_scale="RdYlGn",
                            text="savings_rate", title="Total Spend & Savings Rate by Category",
                            labels={"total_spend": "Total Spend ($)", "savings_rate": "Savings %"})
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(xaxis_tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.pie(fin_summary, names="category", values="cost_savings",
                            hole=0.4, title="Cost Savings Distribution by Category")
                st.plotly_chart(fig, use_container_width=True)

            # Trend over periods
            if not fin_detail.empty and "period" in fin_detail.columns:
                st.subheader("📈 Spend vs Savings Trend by Quarter")
                period_agg = fin_detail.groupby("period").agg(
                    total_spend=("total_spend", "sum"),
                    cost_savings=("cost_savings", "sum")
                ).reset_index().sort_values("period")
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(name="Total Spend", x=period_agg["period"],
                                    y=period_agg["total_spend"], marker_color="#93c5fd"), secondary_y=False)
                fig.add_trace(go.Scatter(name="Cost Savings", x=period_agg["period"],
                                        y=period_agg["cost_savings"], mode="lines+markers",
                                        line=dict(color="#22c55e", width=2)), secondary_y=True)
                fig.update_layout(title="Quarterly Spend vs Savings", height=380)
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 Financial Details by Vendor")
            st.dataframe(fin_detail.round(2), use_container_width=True)
            st.download_button("📥 Download CSV", fin_detail.to_csv(index=False).encode(),
                            "financial_data.csv", "text/csv")

        # ─────────────────────────────────────────────────────────────────────────
        # 4. RISK MANAGEMENT
        # ─────────────────────────────────────────────────────────────────────────
    def render_risk_management(self):
            render_risk_management_page(self)

        # ─────────────────────────────────────────────────────────────────────────
        # 6. COMPLIANCE
        # ─────────────────────────────────────────────────────────────────────────
    def render_compliance(self):
            st.markdown('<div class="main-header">📋 Compliance Management</div>',
                        unsafe_allow_html=True)

            comp = to_df(self.db.get_compliance_data())
            if comp.empty:
                st.warning("No compliance data found.")
                return

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Vendors",   len(comp))
            k2.metric("Compliant",       int((comp["compliance_status"] == "Compliant").sum()))
            k3.metric("Non-Compliant",   int((comp["compliance_status"] == "Non-Compliant").sum()))
            k4.metric("Avg Audit Score", f"{comp['audit_score'].mean():.1f}")

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(comp, names="compliance_status",
                            color="compliance_status",
                            color_discrete_map={"Compliant": "#22c55e",
                                                "Non-Compliant": "#ef4444",
                                                "Under Review": "#f59e0b"},
                            hole=0.4, title="Compliance Status Distribution")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.bar(comp.sort_values("audit_score"),
                            x="audit_score", y="vendor_name", orientation="h",
                            color="compliance_status",
                            color_discrete_map={"Compliant": "#22c55e",
                                                "Non-Compliant": "#ef4444",
                                                "Under Review": "#f59e0b"},
                            title="Audit Scores by Vendor",
                            labels={"audit_score": "Audit Score", "vendor_name": ""})
                st.plotly_chart(fig, use_container_width=True)

            # Upcoming audits
            if "next_audit_date" in comp.columns:
                st.subheader("📅 Upcoming Audits (Next 90 Days)")
                comp["next_audit_date"] = pd.to_datetime(comp["next_audit_date"], errors="coerce")
                upcoming = comp[comp["next_audit_date"] <= datetime.now() + timedelta(days=90)].sort_values("next_audit_date")
                if not upcoming.empty:
                    st.dataframe(upcoming[["vendor_name", "next_audit_date", "compliance_status",
                                        "audit_score", "certifications"]].round(2),
                                use_container_width=True)
                else:
                    st.info("No audits due in the next 90 days.")

            st.subheader("📋 Full Compliance Records")
            st.dataframe(comp.round(2), use_container_width=True)
            st.download_button("📥 Download CSV", comp.to_csv(index=False).encode(),
                            "compliance_data.csv", "text/csv")

        # ─────────────────────────────────────────────────────────────────────────
        # 7. ML PREDICTIONS (NEW & REAL)
        # ─────────────────────────────────────────────────────────────────────────
    def render_ml_predictions(self):
            st.markdown('<div class="main-header">🤖 ML Predictions & Insights</div>',
                        unsafe_allow_html=True)

            if not _ML_AVAILABLE:
                st.error("scikit-learn is not installed. Run: `pip install scikit-learn`")
                return

            ml = self.ml
            if ml is None:
                st.error("ML engine could not be loaded.")
                return

            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 Risk Predictions",
                "📉 Churn Probability",
                "📈 Performance Forecast",
                "🔍 Anomaly Detection",
            ])

            # ── Tab 1: Risk Predictions ──────────────────────────────────────────
            with tab1:
                st.subheader("🎯 Weighted Risk Scoring")
                st.caption(
                    "Transparent weighted score over performance, defect rate and "
                    "on-time delivery — a rule-based screen, not a trained model. "
                    "For the trained churn classifier see the Churn tab."
                )
                with st.spinner("Running risk model…"):
                    risk_pred = to_df(ml.predict_vendor_risks())

                if not risk_pred.empty:
                    # Agreement rate
                    if "risk_level" in risk_pred.columns and "ml_risk_label" in risk_pred.columns:
                        agree = (risk_pred["risk_level"] == risk_pred["ml_risk_label"]).mean() * 100
                        st.metric("Model-DB Agreement Rate", f"{agree:.1f}%",
                                help="% of vendors where ML prediction matches rule-based risk level")

                    col1, col2 = st.columns(2)
                    with col1:
                        ml_dist = risk_pred["ml_risk_label"].value_counts().reset_index()
                        ml_dist.columns = ["Risk Level", "Count"]
                        fig = px.pie(ml_dist, names="Risk Level", values="Count",
                                    color="Risk Level",
                                    color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"},
                                    hole=0.4, title="ML-Predicted Risk Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = px.scatter(risk_pred, x="avg_performance", y="overall_risk",
                                        color="ml_risk_label",
                                        color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"},
                                        hover_name="name", title="Performance vs Risk Score",
                                        labels={"avg_performance": "Avg Performance (%)",
                                                "overall_risk": "Overall Risk Score (%)"})
                        st.plotly_chart(fig, use_container_width=True)

                    # Show probability columns
                    prob_cols = [c for c in risk_pred.columns if c.startswith("prob_")]
                    if prob_cols:
                        st.subheader("🎲 Risk Probability Matrix")
                        show_cols = ["name", "category", "ml_risk_label"] + prob_cols
                        show_cols = [c for c in show_cols if c in risk_pred.columns]
                        st.dataframe(risk_pred[show_cols].round(3), use_container_width=True)

                else:
                    st.info("No risk predictions available.")

                col_retrain, _ = st.columns([1, 4])
                if col_retrain.button("🔁 Retrain Models"):
                    with st.spinner("Retraining…"):
                        ml.retrain()
                    st.success("✅ Models retrained successfully!")

            # ── Tab 2: Churn Probability ─────────────────────────────────────────
            with tab2:
                st.subheader("📉 Vendor Churn Prediction")
                st.caption(
                    "Supervised classifier trained on labelled churn outcomes. "
                    "Features (performance, escalations, financials, risk) come from "
                    "quarter t; the target is churn in quarter t+1 — no leakage. "
                    "Evaluated on held-out recent quarters."
                )
                outcomes = to_df(self.db.get_vendor_outcomes())
                if not outcomes.empty:
                    latest_outcomes = outcomes.sort_values("period").drop_duplicates("vendor_id", keep="last")
                    o1, o2, o3, o4 = st.columns(4)
                    o1.metric("Renewed Contracts", int((latest_outcomes["contract_renewed"] == 1).sum()))
                    o2.metric("Churned Vendors", int((outcomes.groupby("vendor_id")["churned"].max() == 1).sum()))
                    o3.metric("Escalations", int((latest_outcomes["escalation_flag"] == 1).sum()))
                    o4.metric("SLA Breaches", int((latest_outcomes["sla_breach_flag"] == 1).sum()))

                with st.spinner("Training churn model on labelled outcomes…"):
                    predictor, churn_metrics = self._churn_predictor()

                if predictor is None:
                    st.info("Not enough labelled outcome data to train the churn model.")
                else:
                    st.markdown("**Model card** — honest numbers, not vanity metrics:")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Test ROC-AUC", f"{churn_metrics.roc_auc:.3f}",
                              help="Discrimination on held-out quarters; 0.5 = coin flip")
                    m2.metric("CV AUC (train)", f"{churn_metrics.cv_auc_mean:.2f} ± {churn_metrics.cv_auc_std:.2f}",
                              help="GroupKFold by vendor on training quarters")
                    m3.metric("Base churn rate", f"{churn_metrics.base_churn_rate:.1%}",
                              help="Churn is a rare event — accuracy would be misleading")
                    m4.metric("Model", churn_metrics.model_name.replace("_", " ").title())
                    st.caption(
                        "⚠️ Probabilities are class-weight adjusted: use them to *rank* "
                        "vendors, not as calibrated likelihoods."
                    )

                    scored = predictor.predict_current(to_df(self.db.get_vendors()))
                    col1, col2 = st.columns(2)
                    with col1:
                        top = scored.head(15)
                        fig = px.bar(top, x="churn_probability", y="vendor_name", orientation="h",
                                    color="churn_risk",
                                    color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"},
                                    title="Top 15 Vendors by Churn Risk (next quarter)",
                                    labels={"churn_probability": "Churn Score", "vendor_name": ""})
                        fig.update_layout(yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        imp = predictor.feature_importance().head(8)
                        fig = px.bar(imp, x="importance", y="feature", orientation="h",
                                    title="What drives churn risk (feature importance)",
                                    labels={"importance": "Importance", "feature": ""})
                        fig.update_layout(yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig, use_container_width=True)

                    if "value_at_risk" in scored.columns:
                        exposure = scored.head(10)["value_at_risk"].sum()
                        st.warning(
                            f"💰 **Business impact:** the 10 highest-risk vendors represent "
                            f"~${exposure:,.0f} of churn-weighted contract value. "
                            "Proactive QBRs with this list is where the model pays for itself."
                        )

                    st.dataframe(scored.round(3), use_container_width=True)

            # ── Tab 3: Performance Forecast ──────────────────────────────────────
            with tab3:
                st.subheader("📈 6-Month Performance Forecast")
                st.caption(
                    "Holt-Winters exponential smoothing, validated by a rolling-origin "
                    "backtest against a naive (last value) baseline. A forecast only "
                    "earns trust by beating that baseline."
                )
                from core_modules.forecasting import forecast_scores

                perf = to_df(self.db.get_performance_data())
                if perf.empty:
                    st.info("Not enough historical data for forecasts.")
                else:
                    scope = st.selectbox(
                        "Forecast scope",
                        ["Portfolio average"] + sorted(perf["vendor_name"].dropna().unique().tolist()),
                    )
                    vendor = None if scope == "Portfolio average" else scope
                    try:
                        with st.spinner("Backtesting and fitting model…"):
                            result = forecast_scores(perf, vendor_name=vendor, horizon=6)
                    except ValueError as exc:
                        st.info(str(exc))
                    else:
                        b1, b2, b3 = st.columns(3)
                        b1.metric("Backtest MAPE (model)", f"{result.mape_model:.2f}%")
                        b2.metric("Backtest MAPE (naive)", f"{result.mape_naive:.2f}%")
                        b3.metric("Beats naive baseline", "✅ Yes" if result.beats_naive else "❌ No",
                                  help=f"Rolling-origin, {result.backtest_points} one-step folds")

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=result.history["forecast_date"], y=result.history["actual_score"],
                            name="History", mode="lines", line={"color": "#3b82f6"}))
                        fig.add_trace(go.Scatter(
                            x=result.forecast["forecast_date"], y=result.forecast["predicted_score"],
                            name="Forecast", mode="lines+markers", line={"color": "#f59e0b"}))
                        fig.add_trace(go.Scatter(
                            x=pd.concat([result.forecast["forecast_date"],
                                         result.forecast["forecast_date"][::-1]]),
                            y=pd.concat([result.forecast["upper"], result.forecast["lower"][::-1]]),
                            fill="toself", fillcolor="rgba(245,158,11,0.15)",
                            line={"width": 0}, name="95% interval", showlegend=True))
                        fig.update_layout(
                            title=f"{scope}: 6-month forecast ({result.method})",
                            yaxis_title="Performance score")
                        fig.add_hline(y=70, line_dash="dot", line_color="red",
                                    annotation_text="Performance Threshold (70%)")
                        st.plotly_chart(fig, use_container_width=True)

                        st.dataframe(result.forecast.round(2), use_container_width=True, hide_index=True)

            # ── Tab 4: Anomaly Detection ──────────────────────────────────────────
            with tab4:
                st.subheader("🔍 Anomaly Detection")
                st.caption("Isolation Forest detects vendors whose metrics deviate significantly from the norm.")
                with st.spinner("Running anomaly detection…"):
                    anomalies = to_df(ml.detect_anomalies())

                if not anomalies.empty:
                    num_anomalies = int(anomalies["is_anomaly"].sum())
                    k1, k2 = st.columns(2)
                    k1.metric("Anomalous Vendors Detected", num_anomalies)
                    k2.metric("Anomaly Rate", f"{num_anomalies/len(anomalies)*100:.1f}%")

                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.scatter(anomalies, x="avg_performance", y="anomaly_score",
                                        color="is_anomaly",
                                        color_discrete_map={True: "#ef4444", False: "#22c55e"},
                                        hover_name="name",
                                        title="Anomaly Score vs Performance",
                                        labels={"avg_performance": "Avg Performance (%)",
                                                "anomaly_score": "Anomaly Score (lower = more anomalous)"})
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        anom_only = anomalies[anomalies["is_anomaly"]]
                        if not anom_only.empty:
                            fig = px.bar(anom_only, x="name", y="anomaly_score",
                                        color="category", title="Anomalous Vendors",
                                        labels={"anomaly_score": "Anomaly Score", "name": ""})
                            fig.update_layout(xaxis_tickangle=-35)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.success("✅ No anomalies detected!")

                    st.subheader("📋 Anomaly Detection Results")
                    st.dataframe(anomalies.round(3), use_container_width=True)
                else:
                    st.info("No anomaly data available.")

        # ─────────────────────────────────────────────────────────────────────────
        # 8. REPORTS
        # ─────────────────────────────────────────────────────────────────────────
    def render_reports(self):
            render_reports_page(self)

        # ─────────────────────────────────────────────────────────────────────────
        # 9. VENDOR PORTAL
        # ─────────────────────────────────────────────────────────────────────────
    def render_vendor_portal(self):
            st.markdown('<div class="main-header">🏢 Vendor Portal</div>',
                        unsafe_allow_html=True)

            vendors = to_df(self.db.get_vendors())
            tab1, tab2, tab3 = st.tabs(["➕ Add Vendor", "📊 Performance View", "📝 Documents"])

            with tab1:
                st.subheader("Add New Vendor")
                with st.form("add_vendor_form"):
                    c1, c2 = st.columns(2)
                    name = c1.text_input("Company Name *")
                    category = c2.selectbox("Category", ["IT Services", "Logistics", "Manufacturing",
                                                        "Consulting", "Raw Materials", "Marketing", "Facilities"])
                    email = c1.text_input("Email")
                    phone = c2.text_input("Phone")
                    contract_val = c1.number_input("Contract Value ($)", 0, 10_000_000, 50000, 1000)
                    risk_level = c2.selectbox("Initial Risk Level", ["Low", "Medium", "High"])
                    status = c1.selectbox("Status", ["Active", "Inactive", "Under Review"])
                    country = c2.selectbox("Country", ["USA", "UK", "Germany", "India", "Canada", "Australia", "Japan"])
                    submitted = st.form_submit_button("➕ Add Vendor", type="primary")
                    if submitted:
                        if not name:
                            st.error("Company name is required.")
                        else:
                            self.db.add_vendor(name, email, phone, category, status,
                                            risk_level, contract_val, 0, country)
                            st.success(f"✅ Vendor **{name}** added successfully!")
                            st.rerun()

            with tab2:
                st.subheader("Vendor Performance Summary")
                vp = to_df(self.db.get_vendors_with_performance())
                if not vp.empty:
                    sel_vendor = st.selectbox("Select Vendor", vp["name"].unique())
                    v_row = vp[vp["name"] == sel_vendor].iloc[0]
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Avg Performance",  f"{v_row.get('avg_performance', 0):.1f}%")
                    m2.metric("Avg On-Time",       f"{v_row.get('avg_on_time', 0):.1f}%")
                    m3.metric("Avg Quality",       f"{v_row.get('avg_quality', 0):.1f}%")
                    m4.metric("Contract Value",    fmt_currency(v_row.get("contract_value", 0)))

                    # Performance history
                    perf = to_df(self.db.get_performance_data())
                    if not perf.empty and "vendor_name" in perf.columns:
                        v_perf = perf[perf["vendor_name"] == sel_vendor].sort_values("metric_date")
                        if not v_perf.empty:
                            fig = px.line(v_perf, x="metric_date", y="overall_score",
                                        markers=True, title=f"{sel_vendor} — Historical Performance",
                                        labels={"overall_score": "Score (%)", "metric_date": "Date"})
                            fig.update_layout(yaxis=dict(range=[0, 100]))
                            st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.subheader("Document Management")
                docs = pd.DataFrame({
                    "Document": ["Certificate of Insurance", "W-9 / Tax Form",
                                "Quality Certifications", "Compliance Documents", "NDA"],
                    "Status": ["✅ Approved", "✅ Approved", "🔄 Under Review", "✅ Approved", "✅ Approved"],
                    "Last Updated": ["2025-01-15", "2025-01-20", "2025-02-01", "2025-02-28", "2024-12-01"],
                })
                st.dataframe(docs, use_container_width=True)
                st.file_uploader("Upload New Document", type=["pdf", "doc", "docx", "xlsx"])

        # ─────────────────────────────────────────────────────────────────────────
        # 10. SETTINGS
        # ─────────────────────────────────────────────────────────────────────────
    def render_settings(self):
            render_settings_page(self)

        # ─────────────────────────────────────────────────────────────────────────
        # MAIN RUN
        # ─────────────────────────────────────────────────────────────────────────
    def run(self):
            inject_styles()
            last_seen = st.session_state.get("last_activity_at")
            if last_seen and st.session_state.get("user") is not None:
                inactive_minutes = (datetime.now() - last_seen).total_seconds() / 60
                if inactive_minutes > self.config.SESSION_TIMEOUT_MINUTES:
                    st.session_state.user = None
                    st.warning("Session expired due to inactivity. Please sign in again.")
            st.session_state.last_activity_at = datetime.now()
            self.render_sidebar()

            if st.session_state.user is None:
                st.info("👈 Please log in using the sidebar to access the dashboard.")

                st.markdown(f"""
### 🤖 Vendor Insight360  

An intelligent vendor analytics platform delivering real-time visibility, risk intelligence, performance forecasting, and proactive insights to optimize vendor management.

**Key Capabilities**

- 🎯 **Risk Intelligence** — Identify high-risk vendors proactively  
- 📉 **Churn Insights** — Predict potential vendor disengagement  
- 📈 **Performance Forecasting** — Anticipate future performance trends  
- 🔍 **Anomaly Detection** — Detect unusual vendor behavior patterns  

**Demo credentials:** `{self.config.DEMO_ADMIN_USERNAME}` / `{self.config.DEMO_ADMIN_PASSWORD}`
""")
                return

            nav_map = {
                "🏠 Overview":          self.render_overview,
                "📊 Vendor Performance": self.render_vendor_performance,
                "💰 Financial Analytics": self.render_financial_analytics,
                "⚠️ Risk Management":    self.render_risk_management,
                "📋 Compliance":         self.render_compliance,
                "🧠 AI Insights":        self.render_ai_workspace,
                "🤖 ML Predictions":     self.render_ml_predictions,
                "🔬 Analytics Lab":      self.render_analytics_lab,
                "📄 Reports":            self.render_reports,
                "🏢 Vendor Portal":      self.render_vendor_portal,
                "⚙️ Settings":           self.render_settings,
            }
            page = st.session_state.get("selected_nav", "🏠 Overview")
            nav_map.get(page, self.render_overview)()


    # ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dashboard = VendorDashboard()
    dashboard.run()

    
