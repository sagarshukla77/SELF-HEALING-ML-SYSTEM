# HealML — Self-Healing ML Operations Platform

HealML is a safe, end-to-end ML lifecycle prototype: it trains a churn classifier, monitors data quality, drift and observed performance, diagnoses likely causes, retrains challengers, and only promotes candidates through deterministic deployment gates.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Interactive API documentation is at `/docs`.

## Demonstration

1. Train a challenger in the dashboard, then deploy it as the first champion.
2. Run **Healthy** or **Data drift**. Data drift shifts charges and usage, produces PSI/KS or JS drift signals, and creates an incident when thresholds are exceeded.
3. Run **Auto-recover drift**. The system trains on the recent window and promotes only if F1 improves by at least 0.005 and recall is at least 0.70.
4. Inspect the registry, incidents, event history, or `/reports/audit`.

## Implemented capabilities

- Synthetic churn data and CSV upload training
- Logistic Regression, Random Forest, HistGradientBoosting comparisons
- Accuracy, precision, recall, F1, ROC-AUC, PR-AUC and Brier score
- Data quality checks: required schema, missingness, duplicates and age range
- PSI + KS numerical drift and Jensen–Shannon categorical drift
- Ranked deterministic diagnosis and incident storage
- Champion/challenger registry backed by SQLite and joblib artifacts
- Safe deployment, rejection, and explicit rollback APIs
- Scenario simulator: healthy, data drift, quality drift, concept drift, candidate failure
- Audit/event report endpoints, tests, Docker support

This MVP intentionally keeps deployment decisions rule-based; no language model can promote a model.
