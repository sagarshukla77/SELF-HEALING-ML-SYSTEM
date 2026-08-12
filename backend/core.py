from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
DB = ARTIFACTS / "healml.db"
RANDOM_SEED = 42
TARGET = "churn"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def setup() -> None:
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, version TEXT UNIQUE, algorithm TEXT, path TEXT,
            metrics TEXT, features TEXT, status TEXT, parent_id TEXT, trigger_incident TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS incidents (id TEXT PRIMARY KEY, type TEXT, severity TEXT, description TEXT,
            diagnosis TEXT, status TEXT, detected_at TEXT);
        CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, kind TEXT, payload TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS baseline (id INTEGER PRIMARY KEY CHECK(id=1), data TEXT, created_at TEXT);
        """)


def record(kind: str, payload: dict[str, Any]) -> None:
    with db() as con:
        con.execute("INSERT INTO events VALUES (?, ?, ?, ?)", (str(uuid.uuid4()), kind, json.dumps(payload), now()))


def sample_data(rows: int = 1200, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, rows)
    charges = np.clip(rng.normal(65, 25, rows), 18, 150)
    support = rng.poisson(1.7, rows)
    usage = np.clip(rng.normal(55, 21, rows), 0, 100)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], rows, p=[.56, .25, .19])
    payment = rng.choice(["electronic", "card", "bank"], rows, p=[.43, .34, .23])
    age = rng.integers(18, 80, rows)
    logit = -1.5 + .026 * charges + .26 * support - .034 * tenure - .018 * usage + (contract == "month-to-month") * .9
    probability = 1 / (1 + np.exp(-logit))
    return pd.DataFrame({"age": age, "tenure": tenure, "monthly_charges": charges.round(2), "usage_frequency": usage.round(2), "support_calls": support, "contract_type": contract, "payment_method": payment, TARGET: rng.binomial(1, probability)})


def validate_frame(frame: pd.DataFrame, require_target: bool = False) -> dict[str, Any]:
    required = {"age", "tenure", "monthly_charges", "usage_frequency", "support_calls", "contract_type", "payment_method"}
    missing_cols = sorted(required - set(frame.columns))
    if require_target and TARGET not in frame:
        missing_cols.append(TARGET)
    invalid_age = int(((pd.to_numeric(frame.get("age", pd.Series(dtype=float)), errors="coerce") < 0) | (pd.to_numeric(frame.get("age", pd.Series(dtype=float)), errors="coerce") > 120)).sum())
    duplicate_rate = float(frame.duplicated().mean()) if len(frame) else 0
    missing_rate = float(frame.isna().mean().mean()) if len(frame.columns) else 1
    score = max(0.0, 100 - 60 * missing_rate - 20 * duplicate_rate - 20 * bool(missing_cols) - min(20, invalid_age))
    return {"valid": not missing_cols and invalid_age == 0, "rows": len(frame), "missing_columns": missing_cols, "missing_rate": round(missing_rate, 4), "duplicate_rate": round(duplicate_rate, 4), "invalid_age_rows": invalid_age, "score": round(score, 1)}


def features(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    x = frame.drop(columns=[TARGET], errors="ignore")
    numeric = x.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in x.columns if c not in numeric]
    return numeric, categorical


def build_model(frame: pd.DataFrame, algorithm: str) -> Pipeline:
    numeric, categorical = features(frame)
    prep = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric), ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
    models = {"logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED), "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1), "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=.08, random_state=RANDOM_SEED)}
    if algorithm not in models:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return Pipeline([("preprocess", prep), ("model", models[algorithm])])


def metrics(model: Pipeline, frame: pd.DataFrame) -> dict[str, float]:
    x, y = frame.drop(columns=[TARGET]), frame[TARGET].astype(int)
    p = model.predict_proba(x)[:, 1]
    pred = (p >= .5).astype(int)
    return {"accuracy": round(float(accuracy_score(y, pred)), 4), "precision": round(float(precision_score(y, pred, zero_division=0)), 4), "recall": round(float(recall_score(y, pred, zero_division=0)), 4), "f1": round(float(f1_score(y, pred, zero_division=0)), 4), "roc_auc": round(float(roc_auc_score(y, p)), 4), "pr_auc": round(float(average_precision_score(y, p)), 4), "brier": round(float(brier_score_loss(y, p)), 4)}


def score(m: dict[str, float]) -> float:
    return round(.5*m["f1"] + .3*m["recall"] + .2*m["roc_auc"], 4)


def register(model: Pipeline, algorithm: str, frame: pd.DataFrame, m: dict[str, float], status: str = "CANDIDATE", parent: str | None = None, incident: str | None = None) -> dict[str, Any]:
    ident = str(uuid.uuid4()); version = f"model_v{next_version()}"; path = ARTIFACTS / f"{ident}.joblib"
    joblib.dump(model, path)
    item = {"id": ident, "version": version, "algorithm": algorithm, "path": str(path), "metrics": m, "features": frame.drop(columns=[TARGET], errors="ignore").columns.tolist(), "status": status, "parent_id": parent, "trigger_incident": incident, "created_at": now()}
    with db() as con:
        con.execute("INSERT INTO models VALUES (:id,:version,:algorithm,:path,:metrics,:features,:status,:parent_id,:trigger_incident,:created_at)", {**item, "metrics": json.dumps(m), "features": json.dumps(item["features"])})
    return item


def next_version() -> int:
    with db() as con: return int(con.execute("SELECT COUNT(*) FROM models").fetchone()[0]) + 1


def model_row(model_id: str | None = None, production: bool = False) -> dict[str, Any] | None:
    q = "SELECT * FROM models WHERE id=?" if model_id else "SELECT * FROM models WHERE status='PRODUCTION' ORDER BY created_at DESC LIMIT 1"
    with db() as con: row = con.execute(q, (model_id,) if model_id else ()).fetchone()
    if not row: return None
    result = dict(row); result["metrics"] = json.loads(result["metrics"]); result["features"] = json.loads(result["features"]); return result


def deploy(candidate_id: str) -> dict[str, Any]:
    candidate = model_row(candidate_id)
    if not candidate: raise ValueError("Model not found")
    champion = model_row(production=True)
    if candidate["metrics"]["recall"] < .70: return {"approved": False, "reason": "Recall below safety minimum (0.70)"}
    if champion and candidate["metrics"]["f1"] <= champion["metrics"]["f1"] + .005: return {"approved": False, "reason": "F1 did not improve by 0.005"}
    with db() as con:
        con.execute("UPDATE models SET status='ARCHIVED' WHERE status='PRODUCTION'")
        con.execute("UPDATE models SET status='PRODUCTION' WHERE id=?", (candidate_id,))
    record("deployment", {"model_id": candidate_id, "decision": "approved"})
    return {"approved": True, "model": model_row(candidate_id)}


def baseline(frame: pd.DataFrame) -> None:
    payload = frame.drop(columns=[TARGET], errors="ignore").to_json(orient="split")
    with db() as con: con.execute("INSERT OR REPLACE INTO baseline VALUES (1, ?, ?)", (payload, now()))


def get_baseline() -> pd.DataFrame | None:
    with db() as con: row = con.execute("SELECT data FROM baseline WHERE id=1").fetchone()
    return pd.read_json(row[0], orient="split") if row else None


def drift(reference: pd.DataFrame, current: pd.DataFrame) -> list[dict[str, Any]]:
    result = []
    for col in reference.columns:
        if col not in current: continue
        ref, cur = reference[col].dropna(), current[col].dropna()
        if pd.api.types.is_numeric_dtype(ref):
            edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
            if len(edges) < 2: psi = 0.; ks = 0.
            else:
                a, _ = np.histogram(ref, bins=edges); b, _ = np.histogram(cur, bins=edges); aa=(a+1)/(a.sum()+len(a)); bb=(b+1)/(b.sum()+len(b)); psi=float(np.sum((bb-aa)*np.log(bb/aa))); ks=float(ks_2samp(ref, cur).statistic)
            value = max(psi, ks); detail = {"psi": round(psi, 4), "ks": round(ks, 4), "baseline_mean": round(float(ref.mean()), 3), "current_mean": round(float(cur.mean()), 3)}
        else:
            cats = sorted(set(ref.astype(str)) | set(cur.astype(str))); a=np.array([(ref.astype(str)==c).sum() for c in cats])+1; b=np.array([(cur.astype(str)==c).sum() for c in cats])+1; value=float(jensenshannon(a/a.sum(), b/b.sum())); detail={"js_divergence": round(value, 4), "new_categories": sorted(set(cur.astype(str))-set(ref.astype(str)))}
        severity = "HIGH" if value > .25 else "MODERATE" if value > .10 else "LOW"
        result.append({"feature": col, "score": round(value, 4), "severity": severity, **detail})
    return sorted(result, key=lambda x: x["score"], reverse=True)


def diagnose(quality: dict[str, Any], drifts: list[dict[str, Any]], current_metrics: dict[str, float] | None, champion_metrics: dict[str, float] | None) -> dict[str, Any]:
    causes=[]
    for d in drifts[:3]:
        if d["severity"] in {"HIGH", "MODERATE"}: causes.append({"cause": f"Distribution drift in {d['feature']}", "confidence": min(.95, round(.45 + d["score"], 2))})
    if quality["missing_rate"] > .05: causes.append({"cause": "Elevated missing values", "confidence": round(min(.9, .45+quality["missing_rate"]*2),2)})
    if champion_metrics and current_metrics and current_metrics["f1"] < champion_metrics["f1"]*.95: causes.append({"cause": "Concept/performance drift", "confidence": .82})
    return {"ranked_causes": sorted(causes, key=lambda x:x["confidence"], reverse=True), "recommended_strategy": "feature_repair_and_retrain" if quality["missing_rate"] > .08 else "retrain_recent_window" if causes else "observe"}
