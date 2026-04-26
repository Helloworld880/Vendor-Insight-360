try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs): pass
from pathlib import Path
import os
import json
import re
import textwrap
import warnings
from typing import Optional
import pandas as pd
try:
    import anthropic
except ImportError:
    anthropic = None

warnings.filterwarnings(
    "ignore",
    message="urllib3 .* or chardet .*/charset_normalizer .* doesn't match a supported version!",
)
import requests

# ───────────── ENV LOADING ─────────────

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ⭐ AI MODE TOGGLE
# auto   -> free local Ollama first, then mock fallback
# ollama -> Ollama only
# real   -> Anthropic API only
# mock   -> local rule-based demo mode
AI_MODE = os.getenv("AI_MODE", "auto").lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
LAST_AI_BACKEND = "uninitialized"


# ───────────── CLIENT ─────────────
class AIProvider:
    name = "unknown"

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> str:
        raise NotImplementedError


class MockProvider(AIProvider):
    name = "mock"

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> str:
        return _mock_claude_response(user)


class OllamaProvider(AIProvider):
    name = "ollama"

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> str:
        return _call_ollama(system, user)


class AnthropicProvider(AIProvider):
    name = "anthropic"

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> str:
        client = _get_client()
        message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text.strip()


def _get_client():
    if anthropic is None:
        raise ImportError(
            "The 'anthropic' package is not installed. Add it to requirements.txt or use AI_MODE=mock/ollama."
        )
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not found. Add it to your .env file.")
    return anthropic.Anthropic(api_key=api_key)


def _has_anthropic_key() -> bool:
    return anthropic is not None and bool(os.getenv("ANTHROPIC_API_KEY"))


def _call_ollama(system: str, user: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": f"{system}\n\n{user}", "stream": False},
        timeout=45,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


# ───────────── MOCK AI HELPERS ─────────────
def _normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _find_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    normalized = {_normalize_col(col): col for col in df.columns}
    for candidate in candidates:
        match = normalized.get(_normalize_col(candidate))
        if match:
            return match
    return None


def _extract_threshold(text: str, default: float = 70.0) -> float:
    match = re.search(r"(?:below|under|less than)\s+(\d+(?:\.\d+)?)", text.lower())
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if match:
        return float(match.group(1))
    return default


def _extract_prompt_focus(user: str) -> str:
    for marker in ("QUESTION:", "TASK:"):
        if marker in user:
            return user.rsplit(marker, 1)[-1].strip()
    return user.strip()


def _extract_datasets_from_prompt(user: str) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    pattern = re.compile(
        r"--- (?P<label>.+?) DATA ---\n.*?JSON_DATA:\n(?P<json>\[.*?\])\n\nSTATISTICS:",
        re.DOTALL,
    )
    for match in pattern.finditer(user):
        label = match.group("label").strip().lower()
        payload = match.group("json")
        try:
            datasets[label] = pd.DataFrame(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return datasets


def _pick_dataset(datasets: dict[str, pd.DataFrame], required_cols: tuple[str, ...]) -> pd.DataFrame:
    for df in datasets.values():
        if all(_find_column(df, col) for col in required_cols):
            return df.copy()
    for df in datasets.values():
        if any(_find_column(df, col) for col in required_cols):
            return df.copy()
    return pd.DataFrame()


def _vendor_column(df: pd.DataFrame) -> Optional[str]:
    return _find_column(df, "vendor_name", "vendor", "supplier_name", "supplier")


def _currency(value: float) -> str:
    return f"INR {value:,.2f}"


def _percent(value: float) -> str:
    return f"{value:.1f}%"


def _performance_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
    if df.empty:
        return df, None
    vendor_col = _vendor_column(df)
    metric_cols = [
        col for col in (
            _find_column(df, "compliance_score", "compliance"),
            _find_column(df, "on_time_delivery", "on_time_delivery_rate", "delivery_rate"),
            _find_column(df, "quality_score", "quality"),
            _find_column(df, "performance_score", "performance"),
        )
        if col
    ]
    perf_df = df.copy()
    for col in metric_cols:
        perf_df[col] = pd.to_numeric(perf_df[col], errors="coerce")
    if metric_cols:
        perf_df["_composite_score"] = perf_df[metric_cols].mean(axis=1, skipna=True)
    return perf_df, vendor_col


def _financial_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str], Optional[str]]:
    if df.empty:
        return df, None, None
    vendor_col = _vendor_column(df)
    variance_col = _find_column(df, "cost_variance", "cost_overrun", "variance")
    actual_col = _find_column(df, "actual_cost", "actual_spend")
    contract_col = _find_column(df, "contract_value", "budget", "planned_cost")
    fin_df = df.copy()
    for col in (variance_col, actual_col, contract_col):
        if col:
            fin_df[col] = pd.to_numeric(fin_df[col], errors="coerce")
    if not variance_col and actual_col and contract_col:
        variance_col = "_derived_cost_variance"
        fin_df[variance_col] = fin_df[actual_col] - fin_df[contract_col]
    return fin_df, vendor_col, variance_col


def _build_alert_json(user: str) -> str:
    values = {}
    for field in ("VENDOR", "METRIC", "PREVIOUS VALUE", "CURRENT VALUE", "CHANGE", "ALERT THRESHOLD"):
        match = re.search(rf"{re.escape(field)}:\s*(.+)", user)
        if match:
            values[field] = match.group(1).strip()

    vendor_name = values.get("VENDOR", "Vendor")
    metric = values.get("METRIC", "metric")
    previous_value = float(values.get("PREVIOUS VALUE", "0"))
    current_value = float(values.get("CURRENT VALUE", "0"))
    pct_change = float(str(values.get("CHANGE", "0")).replace("%", ""))
    threshold = float(values.get("ALERT THRESHOLD", "0"))
    threshold_gap_pct = 0.0 if threshold == 0 else ((threshold - current_value) / threshold) * 100

    if pct_change <= -20 or threshold_gap_pct > 30:
        severity, urgency = "critical", "immediate"
    elif pct_change <= -10 or threshold_gap_pct > 10:
        severity, urgency = "warning", "within 48 hours"
    else:
        severity, urgency = "info", "this week"

    return json.dumps({
        "severity": severity,
        "subject": f"{metric.title()} alert for {vendor_name}",
        "headline": f"{vendor_name} {metric} moved from {previous_value:g} to {current_value:g}.",
        "explanation": (
            f"The latest {metric} reading is now below the target threshold of {threshold:g}. "
            "This change suggests rising vendor risk and should be reviewed against recent operating performance."
        ),
        "recommendation": f"Review {vendor_name}'s {metric} trend and agree a corrective plan with the vendor owner.",
        "urgency": urgency,
    })


def _answer_data_question(question: str, datasets: dict[str, pd.DataFrame]) -> str:
    perf_df, perf_vendor_col = _performance_dataframe(
        _pick_dataset(datasets, ("compliance_score", "on_time_delivery", "quality_score"))
    )
    fin_df, fin_vendor_col, variance_col = _financial_dataframe(
        _pick_dataset(datasets, ("cost_variance", "actual_cost", "contract_value"))
    )
    question_lower = question.lower()

    if "how many vendor" in question_lower and perf_vendor_col:
        return f"There are {perf_df[perf_vendor_col].nunique()} vendors in the available performance dataset."

    if ("average compliance" in question_lower or "mean compliance" in question_lower) and not perf_df.empty:
        compliance_col = _find_column(perf_df, "compliance_score", "compliance")
        if compliance_col:
            avg = pd.to_numeric(perf_df[compliance_col], errors="coerce").mean()
            return f"The average compliance score is {_percent(avg)}."

    if ("compliance" in question_lower and any(t in question_lower for t in ("below", "under", "less than"))) and not perf_df.empty:
        compliance_col = _find_column(perf_df, "compliance_score", "compliance")
        if compliance_col and perf_vendor_col:
            threshold = _extract_threshold(question)
            filtered = perf_df[pd.to_numeric(perf_df[compliance_col], errors="coerce") < threshold]
            if filtered.empty:
                return f"No vendors are below the {_percent(threshold)} compliance threshold."
            vendors = ", ".join(
                f"{row[perf_vendor_col]} ({_percent(float(row[compliance_col]))})"
                for _, row in filtered.sort_values(compliance_col).iterrows()
            )
            return f"The vendors below {_percent(threshold)} compliance are {vendors}."

    if any(t in question_lower for t in ("highest cost", "cost overrun", "cost escalation", "highest variance")) and not fin_df.empty:
        if variance_col and fin_vendor_col:
            row = fin_df.sort_values(variance_col, ascending=False).iloc[0]
            return f"{row[fin_vendor_col]} has the highest cost variance at {_currency(float(row[variance_col]))}."

    if any(t in question_lower for t in ("top vendor", "best vendor", "top performing", "highest performer")) and not perf_df.empty:
        if "_composite_score" in perf_df.columns and perf_vendor_col:
            row = perf_df.sort_values("_composite_score", ascending=False).iloc[0]
            return f"{row[perf_vendor_col]} is the top performing vendor with a composite score of {_percent(float(row['_composite_score']))}."

    if any(t in question_lower for t in ("at risk", "risk vendor", "lowest performing", "underperforming")) and not perf_df.empty:
        if "_composite_score" in perf_df.columns and perf_vendor_col:
            risk_rows = perf_df.sort_values("_composite_score").head(3)
            vendors = ", ".join(
                f"{row[perf_vendor_col]} ({_percent(float(row['_composite_score']))})"
                for _, row in risk_rows.iterrows()
            )
            return f"The highest-risk vendors are {vendors} based on the lowest combined performance metrics."

    if not perf_df.empty and perf_vendor_col:
        return (
            f"I found data for {perf_df[perf_vendor_col].nunique()} vendors, but the mock AI cannot answer "
            "that question precisely yet. Try asking about compliance thresholds, top vendors, cost variance, or risk."
        )
    return "I could not find enough structured vendor data in the prompt to answer reliably."


def _build_summary(task: str, datasets: dict[str, pd.DataFrame]) -> str:
    perf_df, perf_vendor_col = _performance_dataframe(
        _pick_dataset(datasets, ("compliance_score", "on_time_delivery", "quality_score"))
    )
    fin_df, fin_vendor_col, variance_col = _financial_dataframe(
        _pick_dataset(datasets, ("cost_variance", "actual_cost", "contract_value"))
    )

    if perf_df.empty or not perf_vendor_col or "_composite_score" not in perf_df.columns:
        return "The available data is not sufficient to produce a reliable summary."

    compliance_col = _find_column(perf_df, "compliance_score", "compliance")
    top_row = perf_df.sort_values("_composite_score", ascending=False).iloc[0]
    risk_rows = perf_df.sort_values("_composite_score").head(3)
    low_row = risk_rows.iloc[0]
    task_lower = task.lower()

    if "compliance" in task_lower:
        if not compliance_col:
            return "Compliance data is not available."
        below = perf_df[pd.to_numeric(perf_df[compliance_col], errors="coerce") < 70]
        if below.empty:
            return (
                f"Vendor compliance is currently stable with no vendors below 70%. "
                f"{top_row[perf_vendor_col]} leads overall at {_percent(float(top_row['_composite_score']))}."
            )
        names = ", ".join(
            f"{row[perf_vendor_col]} ({_percent(float(row[compliance_col]))})"
            for _, row in below.sort_values(compliance_col).iterrows()
        )
        return (
            f"Compliance is mixed: {names} are below 70%. "
            f"{low_row[perf_vendor_col]} needs the most urgent attention with an overall score of {_percent(float(low_row['_composite_score']))}."
        )

    if "financial" in task_lower:
        if fin_df.empty or not fin_vendor_col or not variance_col:
            return "Financial data is not available."
        highest = fin_df.sort_values(variance_col, ascending=False).iloc[0]
        return (
            f"{highest[fin_vendor_col]} carries the highest cost variance at {_currency(float(highest[variance_col]))}. "
            f"Operationally, {low_row[perf_vendor_col]} remains the weakest performer at {_percent(float(low_row['_composite_score']))}."
        )

    if "risk" in task_lower:
        risk_text = ", ".join(
            f"{row[perf_vendor_col]} ({_percent(float(row['_composite_score']))})"
            for _, row in risk_rows.iterrows()
        )
        return (
            f"Highest-risk vendors: {risk_text}. "
            f"{low_row[perf_vendor_col]} is the priority intervention case. "
            f"{top_row[perf_vendor_col]} sets the benchmark at {_percent(float(top_row['_composite_score']))}."
        )

    summary_parts = [
        f"Overall performance is stable, led by {top_row[perf_vendor_col]} at {_percent(float(top_row['_composite_score']))}.",
        f"{low_row[perf_vendor_col]} is the most at-risk vendor at {_percent(float(low_row['_composite_score']))}.",
    ]
    if not fin_df.empty and fin_vendor_col and variance_col:
        highest = fin_df.sort_values(variance_col, ascending=False).iloc[0]
        summary_parts.append(f"Highest cost exposure: {highest[fin_vendor_col]} at {_currency(float(highest[variance_col]))}.")
    summary_parts.append("Leadership should focus remediation on the lowest-performing vendors.")
    return " ".join(summary_parts)


def _mock_claude_response(user: str) -> str:
    prompt_focus = _extract_prompt_focus(user)
    datasets = _extract_datasets_from_prompt(user)
    user_lower = prompt_focus.lower()
    if '"severity"' in user or "return valid json" in user_lower:
        return _build_alert_json(user)
    if "summary" in user_lower or "write " in user_lower or "brief" in user_lower:
        return _build_summary(prompt_focus, datasets)
    return _answer_data_question(prompt_focus, datasets)


def _call_claude(system: str, user: str, max_tokens: int = 1024) -> str:
    global LAST_AI_BACKEND
    mode = AI_MODE.lower()
    mock_provider = MockProvider()

    if mode == "mock":
        LAST_AI_BACKEND = mock_provider.name
        return mock_provider.generate(system, user, max_tokens=max_tokens)

    if mode in {"auto", "ollama"}:
        try:
            response = OllamaProvider().generate(system, user, max_tokens=max_tokens)
            LAST_AI_BACKEND = f"ollama:{OLLAMA_MODEL}"
            return response
        except Exception:
            if mode == "ollama":
                raise

    if mode == "real" and _has_anthropic_key():
        try:
            response = AnthropicProvider().generate(system, user, max_tokens=max_tokens)
            LAST_AI_BACKEND = f"anthropic:{ANTHROPIC_MODEL}"
            return response
        except Exception:
            raise

    if mode == "real":
        if anthropic is None:
            raise ImportError("AI_MODE is set to 'real' but the 'anthropic' package is not installed.")
        raise EnvironmentError("AI_MODE is set to 'real' but ANTHROPIC_API_KEY is missing.")

    LAST_AI_BACKEND = mock_provider.name
    return mock_provider.generate(system, user, max_tokens=max_tokens)


# ───────────── HELPERS ─────────────

def _dataframe_to_context(df: pd.DataFrame, max_rows: int = 50) -> str:
    sample = df.head(max_rows)
    stats = df.describe(include="all").to_string()
    return (
        f"COLUMNS: {list(df.columns)}\n\n"
        f"SAMPLE DATA:\n{sample.to_string(index=False)}\n\n"
        f"JSON_DATA:\n{sample.to_json(orient='records')}\n\n"
        f"STATISTICS:\n{stats}"
    )


def _trend_signals(df: pd.DataFrame, score_col: str = "overall_score", date_col: str = "metric_date") -> str:
    """Compute plain-English trend signals from a time-series dataframe."""
    if df.empty or score_col not in df.columns or date_col not in df.columns:
        return "No trend data available."
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, score_col]).sort_values(date_col)
    if len(df) < 2:
        return "Insufficient history for trend analysis."
    first, last = float(df[score_col].iloc[0]), float(df[score_col].iloc[-1])
    delta = last - first
    direction = "improving" if delta > 2 else ("declining" if delta < -2 else "stable")
    return (
        f"Score moved from {first:.1f} to {last:.1f} over {len(df)} periods "
        f"({delta:+.1f} points) — trend is {direction}."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 1 — Ask Your Data  (upgraded: multi-turn, step-by-step reasoning)
# ═══════════════════════════════════════════════════════════════════════════════

class VendorDataChat:
    """
    Natural language interface over vendor DataFrames.
    Upgraded: stronger system prompt with chain-of-thought reasoning,
    richer multi-turn history, and follow-up suggestion generation.
    """

    # ── UPGRADED system prompt ────────────────────────────────────────────────
    SYSTEM_PROMPT = textwrap.dedent("""
        You are a senior procurement data analyst for VendorInsight360, an enterprise 
        vendor performance platform. You have direct access to vendor performance, 
        compliance, financial, and risk data provided in each message.

        Your job is to answer questions precisely, reason through the data step by step,
        and surface insights that a procurement manager or CPO would find immediately 
        actionable.

        RULES:
        1. Always base answers strictly on the provided data — never invent numbers.
        2. Reason step by step before giving your final answer. Show brief working 
           (e.g. "Compliance scores: A=82%, B=61%, C=55% → B and C are below 70%").
        3. Lead with a direct, specific answer. Support it with 1-2 data points.
        4. When listing vendors always include metric values in parentheses.
        5. End every answer with exactly ONE short follow-up suggestion framed as:
           "Follow-up you might ask: <question>" — make it relevant to your answer.
        6. Format numbers cleanly: percentages as X.X%, currency with commas.
        7. Keep responses under 180 words unless a detailed breakdown is requested.
        8. If the data is insufficient, say so clearly and suggest what data would help.
    """).strip()

    def __init__(self, *dataframes: pd.DataFrame, labels: Optional[list[str]] = None):
        if labels and len(labels) != len(dataframes):
            raise ValueError("labels length must match number of dataframes")
        self._context_parts = []
        for i, df in enumerate(dataframes):
            label = labels[i] if labels else f"Dataset {i + 1}"
            self._context_parts.append(
                f"--- {label.upper()} DATA ---\n{_dataframe_to_context(df)}"
            )
        self._context = "\n\n".join(self._context_parts)
        self._history: list[dict] = []

    def ask(self, question: str, use_history: bool = True) -> str:
        """Ask a plain-English question. Returns the AI answer as a string."""
        user_prompt = f"DATA CONTEXT:\n{self._context}\n\nQUESTION: {question}"
        if use_history and self._history:
            prior = "\n".join(
                f"Q: {h['q']}\nA: {h['a']}" for h in self._history[-4:]
            )
            user_prompt = f"PRIOR CONVERSATION (for context only):\n{prior}\n\n{user_prompt}"
        answer = _call_claude(self.SYSTEM_PROMPT, user_prompt, max_tokens=600)
        if use_history:
            self._history.append({"q": question, "a": answer})
        return answer

    def extract_followup(self, answer: str) -> Optional[str]:
        """Parse the follow-up suggestion out of an answer string."""
        match = re.search(r"Follow-up you might ask:\s*(.+)", answer)
        return match.group(1).strip() if match else None

    def reset_history(self):
        self._history = []


def streamlit_chat_widget(chat: VendorDataChat):
    """
    Upgraded Streamlit chat widget with follow-up suggestion chips.
    """
    try:
        import streamlit as st
    except ImportError:
        raise ImportError("streamlit is required for streamlit_chat_widget()")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render prior messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Follow-up chip from last assistant message
    last_followup = None
    for msg in reversed(st.session_state.chat_history):
        if msg["role"] == "assistant":
            last_followup = chat.extract_followup(msg["content"])
            break

    if last_followup:
        if st.button(f"💡 {last_followup}", key="followup_chip", use_container_width=False):
            st.session_state["__ai_suggested_prompt"] = last_followup

    # Input
    if prompt := st.chat_input("Ask about your vendors e.g. 'Which vendors are trending worse?'"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):
                answer = chat.ask(prompt)
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 2 — AI Report Summary Generator  (upgraded: trend-aware, richer prompts)
# ═══════════════════════════════════════════════════════════════════════════════

class ReportSummaryGenerator:
    """
    Generates AI-written executive summaries.
    Upgraded: includes trend direction signals, risk trajectory language,
    and stronger action orientation per audience type.
    """

    SYSTEM_PROMPT = textwrap.dedent("""
        You are a senior analyst writing formal vendor performance summaries for 
        VendorInsight360. Your audience is C-level executives and procurement 
        leadership. Every summary must be grounded in the data provided.

        Writing rules:
        - Use formal, authoritative language. Avoid hedging phrases like "may" or "might".
        - Always include real numbers from the data (percentages, dollar amounts, counts).
        - Structure: 
            Sentence 1 — Overall portfolio status with one headline number.
            Sentences 2-3 — Key findings: name the top and bottom performer with values.
            Sentence 4 — Risk trajectory: is the situation improving, stable, or worsening?
            Sentence 5 — Single, specific recommended action with a named owner or timeline.
        - Never use bullet points. Write in flowing, boardroom-ready paragraphs.
        - Maximum 5 sentences unless explicitly asked for more.
        - Do not invent data. Only reference figures present in the dataset.
        - Use decisive language: "requires immediate escalation", "is the priority 
          intervention case", "leadership must act before the next review cycle".
    """).strip()

    # ── UPGRADED prompts with trend + trajectory language ─────────────────────
    SUMMARY_PROMPTS = {
        "executive": (
            "Write an executive summary of overall vendor performance. "
            "Name the single highest-performing vendor and the single most at-risk vendor with their scores. "
            "Comment on whether the overall portfolio risk trajectory is improving or worsening. "
            "Close with one clear action that the CPO should take before the next board review."
        ),
        "compliance": (
            "Write a compliance-focused summary. "
            "State exactly how many vendors are below the 70% compliance threshold and name them with scores. "
            "Identify the vendor with the fastest-declining compliance trend. "
            "Recommend a specific compliance remediation action with a 30-day deadline framing."
        ),
        "financial": (
            "Write a financial risk summary. "
            "Identify the vendor with the highest cost variance and state the exact amount. "
            "Assess whether overall portfolio cost pressure is rising or contained. "
            "Recommend one financial control action — such as contract renegotiation, spend cap, or audit — "
            "and name which vendor it should target first."
        ),
        "risk": (
            "Write a risk assessment summary. "
            "Rank the top 3 at-risk vendors by combined performance and compliance scores, with values. "
            "For each, give one specific reason they are high risk. "
            "Conclude with a prioritized intervention sequence: which vendor to escalate first and why."
        ),
    }

    def generate(
        self,
        vendor_df: pd.DataFrame,
        period: str = "Current Period",
        financial_df: Optional[pd.DataFrame] = None,
        history_df: Optional[pd.DataFrame] = None,
        summary_type: str = "executive",
    ) -> str:
        """
        Generate an AI executive summary.

        Args:
            vendor_df:     Performance/compliance DataFrame
            period:        Reporting period label e.g. "Q1 2025"
            financial_df:  Optional financial metrics DataFrame
            history_df:    Optional time-series performance DataFrame for trend signals
            summary_type:  One of: executive | compliance | financial | risk
        """
        if summary_type not in self.SUMMARY_PROMPTS:
            raise ValueError(f"summary_type must be one of: {list(self.SUMMARY_PROMPTS.keys())}")

        data_context = f"REPORTING PERIOD: {period}\n\n"
        data_context += f"--- PERFORMANCE DATA ---\n{_dataframe_to_context(vendor_df)}"

        if financial_df is not None:
            data_context += f"\n\n--- FINANCIAL DATA ---\n{_dataframe_to_context(financial_df)}"

        # ── NEW: inject trend signals if history is available ─────────────────
        if history_df is not None and not history_df.empty:
            trend_text = _trend_signals(history_df)
            data_context += f"\n\nPORTFOLIO TREND SIGNAL: {trend_text}"

        task = self.SUMMARY_PROMPTS[summary_type]
        user_prompt = f"{data_context}\n\nTASK: {task}"
        return _call_claude(self.SYSTEM_PROMPT, user_prompt, max_tokens=500)

    def generate_all(
        self,
        vendor_df: pd.DataFrame,
        period: str = "Current Period",
        financial_df: Optional[pd.DataFrame] = None,
        history_df: Optional[pd.DataFrame] = None,
    ) -> dict[str, str]:
        """Generate all 4 summary types at once."""
        return {
            stype: self.generate(vendor_df, period, financial_df, history_df, stype)
            for stype in self.SUMMARY_PROMPTS
        }


def inject_summary_into_report(html_template: str, summaries: dict[str, str]) -> str:
    replacements = {
        "{{AI_EXECUTIVE_SUMMARY}}": summaries.get("executive", ""),
        "{{AI_COMPLIANCE_SUMMARY}}": summaries.get("compliance", ""),
        "{{AI_FINANCIAL_SUMMARY}}": summaries.get("financial", ""),
        "{{AI_RISK_SUMMARY}}": summaries.get("risk", ""),
    }
    for placeholder, content in replacements.items():
        html_template = html_template.replace(placeholder, f"<p>{content}</p>")
    return html_template


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 3 — Smart Alert Explanations  (unchanged, kept intact)
# ═══════════════════════════════════════════════════════════════════════════════

class SmartAlertEngine:
    """Generates contextual, plain-English alert explanations with recommended actions."""

    SYSTEM_PROMPT = textwrap.dedent("""
        You are an intelligent alert system for VendorInsight360.
        Explain anomalies in vendor metrics clearly for procurement managers.

        Response format — always return valid JSON exactly like this:
        {
            "severity": "critical|warning|info",
            "subject": "one-line email subject under 10 words",
            "headline": "one sentence summary of what happened",
            "explanation": "2 sentences explaining what this likely means and why it matters",
            "recommendation": "one specific, actionable recommendation",
            "urgency": "immediate|within 48 hours|this week"
        }

        Severity rules:
        - critical: drop > 20% or value > 30% below threshold
        - warning:  drop 10-20% or value 10-30% below threshold
        - info:     drop < 10% or value just at threshold

        Only return the JSON. No preamble, no explanation outside the JSON.
    """).strip()

    def explain(
        self,
        vendor_name: str,
        metric: str,
        current_value: float,
        previous_value: float,
        threshold: float,
        historical_df: Optional[pd.DataFrame] = None,
    ) -> "AlertResult":
        if previous_value == 0:
            pct_change = 0.0 if current_value == 0 else 100.0
        else:
            pct_change = round(((current_value - previous_value) / previous_value) * 100, 1)

        user_prompt = (
            f"VENDOR: {vendor_name}\n"
            f"METRIC: {metric}\n"
            f"PREVIOUS VALUE: {previous_value}\n"
            f"CURRENT VALUE: {current_value}\n"
            f"CHANGE: {pct_change}%\n"
            f"ALERT THRESHOLD: {threshold}\n"
        )

        if historical_df is not None:
            vendor_history = historical_df[
                historical_df.apply(lambda r: vendor_name.lower() in str(r.values).lower(), axis=1)
            ]
            if not vendor_history.empty:
                user_prompt += f"\nVENDOR HISTORICAL DATA:\n{_dataframe_to_context(vendor_history, max_rows=10)}"

        raw = _call_claude(self.SYSTEM_PROMPT, user_prompt, max_tokens=300)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {
                "severity": "warning",
                "subject": f"Alert: {vendor_name} {metric} dropped",
                "headline": f"{vendor_name} {metric} dropped from {previous_value} to {current_value}.",
                "explanation": f"The {metric} has fallen below the threshold of {threshold}.",
                "recommendation": "Review vendor performance and schedule a check-in call.",
                "urgency": "within 48 hours",
            }

        return AlertResult(
            vendor_name=vendor_name, metric=metric,
            current_value=current_value, previous_value=previous_value,
            pct_change=pct_change, threshold=threshold, **data,
        )

    def batch_explain(
        self, alerts: list[dict], historical_df: Optional[pd.DataFrame] = None
    ) -> list["AlertResult"]:
        results = [self.explain(**alert, historical_df=historical_df) for alert in alerts]
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        return sorted(results, key=lambda r: severity_order.get(r.severity, 3))


class AlertResult:
    SEVERITY_EMOJI = {"critical": "[CRITICAL]", "warning": "[WARNING]", "info": "[INFO]"}

    def __init__(self, vendor_name, metric, current_value, previous_value,
                 pct_change, threshold, severity, subject, headline,
                 explanation, recommendation, urgency):
        self.vendor_name = vendor_name
        self.metric = metric
        self.current_value = current_value
        self.previous_value = previous_value
        self.pct_change = pct_change
        self.threshold = threshold
        self.severity = severity
        self.subject = subject
        self.headline = headline
        self.explanation = explanation
        self.recommendation = recommendation
        self.urgency = urgency

    @property
    def email_subject(self) -> str:
        return f"{self.SEVERITY_EMOJI.get(self.severity, '')} VendorInsight360 Alert: {self.subject}"

    @property
    def email_body(self) -> str:
        return textwrap.dedent(f"""
            VendorInsight360 - Automated Alert
            {'=' * 40}
            Vendor  : {self.vendor_name}
            Metric  : {self.metric}
            Change  : {self.previous_value} -> {self.current_value} ({self.pct_change:+.1f}%)
            Severity: {self.severity.upper()}
            Urgency : {self.urgency}

            WHAT HAPPENED
            {self.headline}

            ANALYSIS
            {self.explanation}

            RECOMMENDED ACTION
            {self.recommendation}

            {'-' * 40}
            This alert was generated automatically by VendorInsight360.
        """).strip()

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "vendor_name", "metric", "current_value", "previous_value",
            "pct_change", "severity", "subject", "headline",
            "explanation", "recommendation", "urgency",
        )}

    def __repr__(self):
        return f"AlertResult(vendor={self.vendor_name!r}, metric={self.metric!r}, severity={self.severity!r})"


# ═══════════════════════════════════════════════════════════════════════════════
# NEW FEATURE 4 — Executive Brief Builder
# Audience-aware, structured, board-ready briefs with named sections
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutiveBriefBuilder:
    """
    Generates structured, audience-aware executive briefs with named sections.

    Audience types: "board", "procurement", "operations"
    Tone types: "formal", "direct", "operational"

    Returns a BriefResult with individual sections for flexible rendering.

    Usage:
        builder = ExecutiveBriefBuilder()
        brief = builder.build(
            vendor_df=perf_df,
            review_df=review_df,
            period="May 2025",
            audience="board",
            tone="formal",
            financial_df=fin_df,
            history_df=perf_history,
        )
        print(brief.situation)
        print(brief.key_findings)
        print(brief.risk_outlook)
        print(brief.recommended_actions)
        print(brief.as_text())  # full plain-text for download
    """

    # ── Audience-specific instruction overlays ────────────────────────────────
    AUDIENCE_OVERLAYS = {
        "board": (
            "This brief is for a board of directors. Focus on financial exposure, "
            "strategic risk, and governance implications. Use formal register. "
            "Avoid operational jargon. Quantify impact in dollar terms where possible."
        ),
        "procurement": (
            "This brief is for the Chief Procurement Officer and procurement leads. "
            "Focus on vendor performance metrics, contract risk, compliance gaps, "
            "and actionable sourcing decisions. Be specific about vendor names and scores."
        ),
        "operations": (
            "This brief is for operations and delivery leadership. "
            "Focus on SLA risk, delivery reliability, capacity concerns, and escalation needs. "
            "Use plain, direct language. Name the vendors that need hands-on intervention."
        ),
    }

    TONE_OVERLAYS = {
        "formal": "Write in formal, board-ready prose. No contractions. Authoritative tone.",
        "direct": "Write directly and concisely. Short sentences. State the problem, then the action.",
        "operational": "Write in plain operational language. Focus on what needs to happen next week.",
    }

    SYSTEM_PROMPT = textwrap.dedent("""
        You are a senior strategy analyst writing structured executive briefs for 
        VendorInsight360. You must return your response as valid JSON with exactly 
        these four keys:

        {
          "situation": "1-2 sentences: current portfolio status with key headline number",
          "key_findings": "2-3 sentences: top and bottom performers named with values, compliance/cost highlights",
          "risk_outlook": "1-2 sentences: is portfolio risk improving, stable, or worsening and why",
          "recommended_actions": "2-3 specific, named actions the audience should take, each on its own sentence"
        }

        Rules:
        - Every section must reference real numbers from the data.
        - Never invent data. If a number is not in the data, omit it.
        - Recommended actions must name specific vendors or metrics, not vague guidance.
        - Only return the JSON object. No preamble, no markdown outside the JSON.
    """).strip()

    def build(
        self,
        vendor_df: pd.DataFrame,
        review_df: pd.DataFrame,
        period: str = "Current Period",
        audience: str = "board",
        tone: str = "formal",
        financial_df: Optional[pd.DataFrame] = None,
        history_df: Optional[pd.DataFrame] = None,
    ) -> "BriefResult":
        audience_note = self.AUDIENCE_OVERLAYS.get(audience, self.AUDIENCE_OVERLAYS["board"])
        tone_note = self.TONE_OVERLAYS.get(tone, self.TONE_OVERLAYS["formal"])

        data_context = f"REPORTING PERIOD: {period}\n\n"
        data_context += f"--- PERFORMANCE & RISK REVIEW DATA ---\n{_dataframe_to_context(review_df)}"

        if financial_df is not None and not financial_df.empty:
            data_context += f"\n\n--- FINANCIAL DATA ---\n{_dataframe_to_context(financial_df)}"

        if history_df is not None and not history_df.empty:
            trend_text = _trend_signals(history_df)
            data_context += f"\n\nPORTFOLIO TREND SIGNAL: {trend_text}"

        user_prompt = (
            f"AUDIENCE CONTEXT: {audience_note}\n"
            f"TONE: {tone_note}\n\n"
            f"{data_context}\n\n"
            f"TASK: Write a structured executive brief for the {audience} audience "
            f"covering the current vendor portfolio situation as of {period}."
        )

        raw = _call_claude(self.SYSTEM_PROMPT, user_prompt, max_tokens=700)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Graceful fallback
            data = {
                "situation": "Portfolio data has been loaded but the AI response could not be parsed. Please retry.",
                "key_findings": "Check that ANTHROPIC_API_KEY is set and AI_MODE=real.",
                "risk_outlook": "Risk data is available in the review queue.",
                "recommended_actions": "Review the Risk Management tab for manual analysis.",
            }

        return BriefResult(
            period=period,
            audience=audience,
            tone=tone,
            situation=data.get("situation", ""),
            key_findings=data.get("key_findings", ""),
            risk_outlook=data.get("risk_outlook", ""),
            recommended_actions=data.get("recommended_actions", ""),
        )


class BriefResult:
    """Structured result from ExecutiveBriefBuilder.build()"""

    def __init__(self, period: str, audience: str, tone: str,
                 situation: str, key_findings: str, risk_outlook: str,
                 recommended_actions: str):
        self.period = period
        self.audience = audience
        self.tone = tone
        self.situation = situation
        self.key_findings = key_findings
        self.risk_outlook = risk_outlook
        self.recommended_actions = recommended_actions

    def as_text(self) -> str:
        return textwrap.dedent(f"""
            VendorInsight360 — Executive Brief
            Audience : {self.audience.title()}
            Period   : {self.period}
            Tone     : {self.tone.title()}
            {'=' * 50}

            SITUATION
            {self.situation}

            KEY FINDINGS
            {self.key_findings}

            RISK OUTLOOK
            {self.risk_outlook}

            RECOMMENDED ACTIONS
            {self.recommended_actions}

            {'=' * 50}
            Generated by VendorInsight360 AI Engine
        """).strip()

    def as_dict(self) -> dict:
        return {
            "period": self.period,
            "audience": self.audience,
            "tone": self.tone,
            "situation": self.situation,
            "key_findings": self.key_findings,
            "risk_outlook": self.risk_outlook,
            "recommended_actions": self.recommended_actions,
        }

    def __repr__(self):
        return f"BriefResult(audience={self.audience!r}, period={self.period!r})"


# ═══════════════════════════════════════════════════════════════════════════════
# NEW FEATURE 5 — Vendor Narrative Engine
# One-click AI health paragraph per vendor for risk review drill-downs
# ═══════════════════════════════════════════════════════════════════════════════

class VendorNarrativeEngine:
    """
    Generates a concise, plain-English health narrative for a single vendor.
    Designed for the Risk Review drill-down in the AI Insights tab.

    Usage:
        engine = VendorNarrativeEngine()
        narrative = engine.narrate(vendor_row, peer_avg_dict, history_df)
        print(narrative)
    """

    SYSTEM_PROMPT = textwrap.dedent("""
        You are a procurement risk analyst writing a single-vendor health assessment 
        for VendorInsight360. Your narrative will appear in a risk review dashboard 
        and be read by a procurement manager in 30 seconds.

        Write exactly 3 short paragraphs:
        Paragraph 1 — Current health: summarise performance, compliance, and risk scores 
                       with actual values. State whether this vendor is above or below 
                       peer averages.
        Paragraph 2 — Risk drivers: identify the 1-2 specific factors driving concern 
                       (e.g. declining compliance, cost overrun, operational risk score).
        Paragraph 3 — Recommended intervention: one specific action the vendor owner 
                       should take in the next 2 weeks, and what "good" looks like 
                       (i.e. what metric needs to move and by how much).

        Rules:
        - Use real numbers from the data only.
        - Be direct. No hedging. No bullet points. Pure prose.
        - Maximum 120 words total across all 3 paragraphs.
    """).strip()

    def narrate(
        self,
        vendor_row: dict,
        peer_averages: Optional[dict] = None,
        history_df: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Generate a health narrative for one vendor.

        Args:
            vendor_row:    dict (or pd.Series row) with vendor metrics
            peer_averages: dict with keys like 'avg_performance', 'avg_compliance', 
                           'avg_risk' for peer comparison
            history_df:    Optional performance history for this vendor

        Returns:
            Plain-text 3-paragraph narrative string.
        """
        vendor_name = vendor_row.get("vendor_name", "This vendor")
        user_prompt = f"VENDOR: {vendor_name}\n\nMETRICS:\n"
        for k, v in vendor_row.items():
            if k != "vendor_name" and v is not None:
                user_prompt += f"  {k}: {v}\n"

        if peer_averages:
            user_prompt += "\nPEER AVERAGES:\n"
            for k, v in peer_averages.items():
                user_prompt += f"  {k}: {v:.1f}\n"

        if history_df is not None and not history_df.empty:
            trend = _trend_signals(history_df)
            user_prompt += f"\nPERFORMANCE TREND: {trend}\n"

        return _call_claude(self.SYSTEM_PROMPT, user_prompt, max_tokens=300)


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TEST — python ai_integration.py
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import io

    PERF_CSV = """vendor_name,compliance_score,on_time_delivery,quality_score,month
Vendor A,85,92,88,2025-01
Vendor B,62,74,70,2025-01
Vendor C,91,95,93,2025-01
Vendor D,55,68,60,2025-01
Vendor E,78,85,82,2025-01"""

    FIN_CSV = """vendor_name,contract_value,actual_cost,cost_variance,month
Vendor A,100000,98000,-2000,2025-01
Vendor B,80000,95000,15000,2025-01
Vendor C,120000,118000,-2000,2025-01
Vendor D,60000,72000,12000,2025-01
Vendor E,90000,91000,1000,2025-01"""

    perf_df = pd.read_csv(io.StringIO(PERF_CSV))
    fin_df  = pd.read_csv(io.StringIO(FIN_CSV))

    print(f"AI mode: {AI_MODE}\n")

    print("=" * 60)
    print("FEATURE 1 — Ask Your Data (multi-turn + follow-up chips)")
    print("=" * 60)
    chat = VendorDataChat(perf_df, fin_df, labels=["performance", "financial"])
    q1 = "Which vendors have compliance below 70%?"
    print(f"\nQ: {q1}")
    a1 = chat.ask(q1)
    print(f"A: {a1}")
    followup = chat.extract_followup(a1)
    if followup:
        print(f"\n💡 Follow-up chip: {followup}")
    print(f"Backend: {LAST_AI_BACKEND}")

    print("\n" + "=" * 60)
    print("FEATURE 2 — Report Summary (trend-aware)")
    print("=" * 60)
    gen = ReportSummaryGenerator()
    summary = gen.generate(perf_df, period="Q1 2025", financial_df=fin_df, summary_type="executive")
    print(f"\n{summary}")
    print(f"Backend: {LAST_AI_BACKEND}")

    print("\n" + "=" * 60)
    print("NEW FEATURE 4 — Executive Brief Builder")
    print("=" * 60)
    builder = ExecutiveBriefBuilder()
    brief = builder.build(
        vendor_df=perf_df,
        review_df=perf_df,
        period="Q1 2025",
        audience="board",
        tone="formal",
        financial_df=fin_df,
    )
    print(brief.as_text())
    print(f"Backend: {LAST_AI_BACKEND}")

    print("\n" + "=" * 60)
    print("NEW FEATURE 5 — Vendor Narrative Engine")
    print("=" * 60)
    engine = VendorNarrativeEngine()
    vendor_row = {
        "vendor_name": "Vendor B",
        "compliance_score": 62,
        "on_time_delivery": 74,
        "quality_score": 70,
        "overall_risk": 71,
        "cost_variance": 15000,
        "risk_level": "High",
    }
    peer_avg = {"avg_performance": 78.0, "avg_compliance": 74.2, "avg_risk": 55.0}
    narrative = engine.narrate(vendor_row, peer_avg)
    print(f"\n{narrative}")
    print(f"Backend: {LAST_AI_BACKEND}")
