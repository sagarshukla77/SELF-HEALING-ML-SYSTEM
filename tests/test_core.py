from backend import core

def test_quality_and_drift():
    base = core.sample_data(350, 1)
    current = base.drop(columns=[core.TARGET]).copy(); current["monthly_charges"] += 45
    quality = core.validate_frame(base, True)
    drifts = core.drift(base.drop(columns=[core.TARGET]), current)
    assert quality["valid"]
    assert drifts[0]["severity"] in {"MODERATE", "HIGH"}

def test_training_metrics():
    data=core.sample_data(350, 2); model=core.build_model(data,"logistic_regression")
    model.fit(data.drop(columns=[core.TARGET]), data[core.TARGET])
    values=core.metrics(model,data)
    assert 0 <= values["f1"] <= 1
