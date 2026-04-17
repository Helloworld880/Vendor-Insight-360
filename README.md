<!-- # Vendor-Insight-360

A Streamlit-based vendor analytics platform for tracking **performance**, **financials**, **risk**, and **compliance**, with automation scripts for **alerting** and **scheduled reporting**. Includes a realistic demo dataset generator to keep the dashboard outputs believable.

---

## Tech Stack

- Python
- Streamlit
- SQLite (local DB)
- Plotly / Matplotlib (charts)
- Pytest (tests)

---

## What this app does

- **Vendor Performance**: KPIs + trends across vendors
- **Financial Analytics**: spend/variance/overdues/ROI-style signals
- **Risk Management**: portfolio view + drill-down + trend movement
- **Compliance**: audit score/status tracking
- **Reports**: generate **PDF / Excel / HTML** outputs
- **AI features** (LLM-assisted, with safe fallback):
  - Ask questions over your vendor datasets (“Ask Data”)
  - Generate executive briefs / summaries
  - Explain alerts in plain English with recommended actions
- **Automation**:
  - Alert monitoring (threshold breaches)
  - Scheduled report generation loop

  
## Project Structure (full)

```
├── app.py
├── run.py
├── run_api.py
├── setup.py
├── requirements.txt
├── pytest.ini
├── create_dataset.bat
├── DATASET_IMPROVEMENT_PLAN.md

├── tests/
│   └── test_data_health.py

├── core_modules/
│   ├── __init__.py
│   ├── analytics.py
│   ├── api.py
│   ├── auth.py
│   ├── config.py
│   ├── database.py
│   ├── email_service.py
│   ├── import_dataset.py
│   ├── ml_engine.py
│   ├── realistic_dataset.py
│   ├── risk_model.py
│   ├── vendor_clustering.py
│   └── ... (other helper modules)

├── enhancements/
│   ├── benchmarking.py
│   ├── ml_engine.py
│   ├── report_generator.py
│   └── ... (optional/advanced modules)

├── ui_pages/
│   ├── __init__.py
│   ├── ai_page.py
│   ├── reports_page.py
│   ├── risk_page.py
│   └── settings_page.py

├── data_layer/
│   ├── vendors.csv
│   ├── performance.csv
│   ├── financial_metrics.csv
│   ├── risk_history.csv
│   ├── compliance_history.csv
│   ├── vendor_outcomes.csv
│   ├── industry_benchmarks.csv
│   └── vendors.db

├── workflows_automation/
│   ├── scripts/
│   │   ├── alert_monitor.py
│   │   └── report_scheduler.py
│   └── workflows/
│       ├── issue_escalation.yaml
│       ├── performance_review.yaml
│       └── vendor_onboarding.yaml

├── reports/
└── generated_reports/
---

## Setup

Install dependencies:

<<<<<<< HEAD
```
git clone https://github.com/Helloworld880/Vendor-Insight-360.git
```

Navigate to the project directory

```
cd Vendor-Insight-360
```

Install dependencies

```
=======
```bash
>>>>>>> a7071ca (update)
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

---

## Realistic demo dataset (recommended)

Generate/overwrite realistic CSVs in `Data layer/`:

```bash
python -c "from core_modules.realistic_dataset import DatasetSpec, write_to_data_layer; print(write_to_data_layer('Data layer', DatasetSpec(n_vendors=120, months=24, start_month='2024-01-01', seed=42), overwrite=True))"
```

You can also regenerate from the app via **Settings → Re-seed Database** (this refreshes the demo CSVs too).

---

## Automation scripts

Alert monitor (safe dry run):

```bash
python "WORKFLOWS & AUTOMATION/scripts/alert_monitor.py" --dry-run
```

Report scheduler (runs continuously; supports daily `08:00` and weekly `monday 09:00` patterns):

```bash
python "WORKFLOWS & AUTOMATION/scripts/report_scheduler.py" --run
```

---

## Reports output

Generated reports are stored under:

```text
reports/
generated_reports/
```

---

## Author

Yash Dudhani  
GitHub: `https://github.com/Helloworld880`

---

## License

Educational and research use. -->




# Vendor Insight360 (Vendor Optimization Platform)

A Streamlit-based vendor analytics platform for tracking **performance**, **financials**, **risk**, and **compliance**, with automation scripts for **alerting** and **scheduled reporting**. Includes a realistic demo dataset generator to keep the dashboard outputs believable.

---

## Tech Stack

- Python
- Streamlit
- SQLite (local DB)
- Plotly / Matplotlib (charts)
- Pytest (tests)

---

## What this app does

- **Vendor Performance**: KPIs + trends across vendors
- **Financial Analytics**: spend/variance/overdues/ROI-style signals
- **Risk Management**: portfolio view + drill-down + trend movement
- **Compliance**: audit score/status tracking
- **Reports**: generate **PDF / Excel / HTML** outputs
- **AI features**:
  - Ask questions over your vendor datasets
  - Generate executive summaries
  - Explain alerts with recommendations
- **Automation**:
  - Alert monitoring
  - Scheduled report generation

---

## Project Structure
├── app.py
├── create_dataset.bat
├── DATASET_IMPROVEMENT_PLAN.md
├── pytest.ini
├── requirements.txt
├── run.py
├── run_api.py
├── setup.py
├── tests/
│   └── test_data_health.py
├── core_modules/
│   ├── __init__.py
│   ├── analytics.py
│   ├── api.py
│   ├── auth.py
│   ├── config.py
│   ├── database.py
│   ├── email_service.py
│   ├── import_dataset.py
│   ├── ml_engine.py
│   ├── realistic_dataset.py
│   ├── risk_model.py
│   ├── vendor_clustering.py
│   └── ... (other helpers)
├── enhancements/
│   ├── benchmarking.py
│   ├── ml_engine.py
│   ├── report_generator.py
│   └── ... (optional modules)
├── ui_pages/
│   ├── __init__.py
│   ├── ai_page.py
│   ├── reports_page.py
│   ├── risk_page.py
│   └── settings_page.py
├── Data layer/
│   ├── vendors.csv
│   ├── performance.csv
│   ├── financial_metrics.csv
│   ├── risk_history.csv
│   ├── compliance_history.csv
│   ├── vendor_outcomes.csv
│   ├── industry_benchmarks.csv
│   └── vendors.db
├── WORKFLOWS & AUTOMATION/
│   ├── scripts/
│   │   ├── alert_monitor.py
│   │   └── report_scheduler.py
│   └── workflows/
│       ├── issue_escalation.yaml
│       ├── performance_review.yaml
│       └── vendor_onboarding.yaml
├── reports/
└── generated_reports/



---

## Setup

```bash
git clone https://github.com/Helloworld880/Vendor-Insight-360.git
cd Vendor-Insight-360
pip install -r requirements.txt
streamlit run app.py


Author

Yash Dudhani
GitHub: https://github.com/Helloworld880
