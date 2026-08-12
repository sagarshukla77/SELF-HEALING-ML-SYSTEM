from __future__ import annotations
import io
from pathlib import Path
from typing import Literal
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from backend import core

app = FastAPI(title="HealML", version="1.0.0", description="Safe, deterministic self-healing ML operations prototype")
core.setup()

class TrainRequest(BaseModel):
    algorithm: Literal["logistic_regression", "random_forest", "hist_gradient_boosting"] = "random_forest"
    rows: int = Field(1200, ge=300, le=10000)

class SimulationRequest(BaseModel):
    scenario: Literal["healthy", "data_drift", "quality_drift", "concept_drift", "candidate_failure"] = "data_drift"
    rows: int = Field(350, ge=100, le=5000)

def ensure_champion() -> dict:
    champion = core.model_row(production=True)
    if not champion: raise HTTPException(409, "Train and deploy an initial champion first.")
    return champion

@app.get("/")
def dashboard(): return FileResponse(Path(__file__).parent / "static" / "index.html")

@app.get("/health")
def service_health(): return {"service": "HealML", "status": "ok"}

@app.post("/models/train")
def train(request: TrainRequest):
    data=core.sample_data(request.rows); quality=core.validate_frame(data, True)
    model=core.build_model(data, request.algorithm); model.fit(data.drop(columns=[core.TARGET]), data[core.TARGET]); m=core.metrics(model, data)
    item=core.register(model, request.algorithm, data, m); core.baseline(data)
    return {"model": item, "quality": quality, "selection_score": core.score(m)}

@app.post("/models/upload-train")
async def upload_train(file: UploadFile = File(...), algorithm: str = "random_forest"):
    if not file.filename.endswith(".csv"): raise HTTPException(400, "Only CSV is supported by this endpoint.")
    data=pd.read_csv(io.BytesIO(await file.read())); quality=core.validate_frame(data, True)
    if not quality["valid"]: raise HTTPException(422, quality)
    model=core.build_model(data, algorithm); model.fit(data.drop(columns=[core.TARGET]), data[core.TARGET]); m=core.metrics(model, data); item=core.register(model, algorithm, data, m); core.baseline(data)
    return {"model": item, "quality": quality}

@app.get("/models")
def models():
    with core.db() as con: rows=con.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
    return [{**dict(r), "metrics": __import__('json').loads(r["metrics"]), "features": __import__('json').loads(r["features"])} for r in rows]

@app.post("/models/{model_id}/deploy")
def deploy(model_id: str):
    try: return core.deploy(model_id)
    except ValueError as e: raise HTTPException(404, str(e))

@app.post("/models/{model_id}/rollback")
def rollback(model_id: str):
    target=core.model_row(model_id)
    if not target: raise HTTPException(404, "Model not found")
    with core.db() as con:
        con.execute("UPDATE models SET status='ROLLED_BACK' WHERE status='PRODUCTION'"); con.execute("UPDATE models SET status='PRODUCTION' WHERE id=?", (model_id,))
    core.record("rollback", {"restored_model": model_id}); return {"rolled_back_to": core.model_row(model_id)}

def simulated(request: SimulationRequest) -> pd.DataFrame:
    data=core.sample_data(request.rows, seed=99)
    if request.scenario == "data_drift": data["monthly_charges"] += 38; data["usage_frequency"] = (data["usage_frequency"]*.55).clip(0,100)
    if request.scenario == "quality_drift": data.loc[data.sample(frac=.18, random_state=1).index, "monthly_charges"] = None
    if request.scenario in {"concept_drift", "candidate_failure"}:
        p=.12 + .012*data["tenure"] + .006*data["usage_frequency"] - .008*data["monthly_charges"]
        data[core.TARGET]=(p.clip(.02,.92) > .47).astype(int)
    return data

@app.post("/monitoring/simulate")
def monitor(request: SimulationRequest):
    champion=ensure_champion(); data=simulated(request); reference=core.get_baseline(); quality=core.validate_frame(data, True); drifts=core.drift(reference, data.drop(columns=[core.TARGET]))
    model=__import__('joblib').load(champion["path"]); observed=core.metrics(model, data); diagnosis=core.diagnose(quality, drifts, observed, champion["metrics"])
    degraded=observed["f1"] < champion["metrics"]["f1"]*.95; severe=any(d["severity"]=="HIGH" for d in drifts) or quality["score"]<80
    incident=None
    if degraded or severe:
        incident={"id": f"INC-{__import__('uuid').uuid4().hex[:8].upper()}", "type": "performance_degradation" if degraded else "data_quality_or_drift", "severity": "HIGH" if degraded or severe else "WARNING", "description": f"Scenario {request.scenario}: monitoring threshold breached", "diagnosis": diagnosis, "status": "OPEN", "detected_at": core.now()}
        with core.db() as con: con.execute("INSERT INTO incidents VALUES (:id,:type,:severity,:description,:diagnosis,:status,:detected_at)", {**incident,"diagnosis":__import__('json').dumps(diagnosis)})
    health=round(max(0, min(100, 40*observed["f1"] + .2*quality["score"] + 20*(1-min(1,max([d["score"] for d in drifts], default=0))) + 10*(1-abs(champion["metrics"]["f1"]-observed["f1"])))) ,1)
    core.record("monitoring", {"scenario":request.scenario,"health":health,"incident":incident and incident["id"]})
    return {"health_score":health,"quality":quality,"drift":drifts,"champion_metrics":champion["metrics"],"observed_metrics":observed,"incident":incident,"diagnosis":diagnosis}

@app.post("/recovery/start")
def recovery(request: SimulationRequest):
    champion=ensure_champion(); data=simulated(request); algorithm="hist_gradient_boosting" if request.scenario != "candidate_failure" else "logistic_regression"
    # The failure scenario deliberately trains on stale data but evaluates on the
    # current window.  This demonstrates that the deployment gate rejects a
    # plausible-looking recovery when it cannot generalize to current traffic.
    training_data = core.sample_data(request.rows, seed=7) if request.scenario == "candidate_failure" else data
    model=core.build_model(training_data, algorithm); model.fit(training_data.drop(columns=[core.TARGET]),training_data[core.TARGET]); m=core.metrics(model,data); candidate=core.register(model,algorithm,training_data,m,parent=champion["id"])
    decision=core.deploy(candidate["id"])
    core.record("recovery", {"strategy":"retrain_recent_window","candidate":candidate["id"],"decision":decision})
    return {"strategy":"retrain_recent_window","champion":champion,"candidate":candidate,"validation":decision}

@app.get("/monitoring/health")
def health():
    champion=core.model_row(production=True)
    with core.db() as con: incidents=con.execute("SELECT COUNT(*) FROM incidents WHERE status='OPEN'").fetchone()[0]
    return {"champion":champion,"open_incidents":incidents,"message":"Train a model and use the simulator to calculate live health." if not champion else "Champion available"}

@app.get("/incidents")
def incidents():
    with core.db() as con: rows=con.execute("SELECT * FROM incidents ORDER BY detected_at DESC").fetchall()
    return [{**dict(r),"diagnosis":__import__('json').loads(r["diagnosis"])} for r in rows]

@app.get("/reports/audit")
def audit():
    return {"models":models(),"incidents":incidents(),"events": events()}

@app.get("/events")
def events():
    with core.db() as con: rows=con.execute("SELECT * FROM events ORDER BY created_at DESC LIMIT 100").fetchall()
    return [{**dict(r),"payload":__import__('json').loads(r["payload"])} for r in rows]
