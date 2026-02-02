import pytest
from fastapi.testclient import TestClient
import numpy as np
import joblib

from app.main import app


class DummyPipe:
    def predict_proba(self, df):
        # sklearn-like shape (n_samples, 2)
        return np.array([[0.2, 0.8]], dtype=float)


@pytest.fixture()
def client(monkeypatch):
    # Lifespan в app.main грузит pipeline через joblib.load -> подменяем его
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: DummyPipe())
    with TestClient(app) as c:
        # Подкладываем drift reference вручную (в тестах нет настоящего manifest.json на диске)
        app.state.manifest = {
            "model_version": "test",
            "data": {
                "profile": {
                    "drift_reference": {
                        "features": {
                            "numeric": {
                                "tenure": {"mean": 10.0, "std": 5.0, "edges": [0.0, 5.0, 10.0, 20.0], "ref_bin_probs": [0.25, 0.5, 0.25]},
                                "MonthlyCharges": {"mean": 50.0, "std": 10.0, "edges": [0.0, 40.0, 60.0, 100.0], "ref_bin_probs": [0.25, 0.5, 0.25]},
                            },
                            "categorical": {
                                "Contract": {"freq": {"Month-to-month": 0.6, "Two year": 0.4}},
                                "PaymentMethod": {"freq": {"Electronic check": 0.7, "Mailed check": 0.3}},
                            },
                            "meta": {"rows": 1000, "numeric_bins": 10, "max_categories": 50},
                        },
                        "prediction_proba": {"mean": 0.3, "std": 0.1, "edges": [0.0, 0.2, 0.4, 1.0]},
                    }
                }
            },
        }
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["pipeline_loaded"] is True
    assert data["status"] == "ok"


def test_predict_ok_minimal_payload(client):
    payload = {
        "customer": {
            # можно передать минимально необходимое (остальное Optional)
            "tenure": 1,
            "MonthlyCharges": 10.0,
            "TotalCharges": "10.0",
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
        }
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    # проверяем именно значение из DummyPipe
    assert abs(data["churn_probability"] - 0.8) < 1e-9


def test_predict_when_pipeline_not_loaded(monkeypatch):
    # Симулируем ситуацию, когда joblib.load упал -> lifespan выставит app.state.pipe = None
    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(joblib, "load", _raise)
    with TestClient(app) as client:
        r = client.post("/predict", json={"customer": {}})
        assert r.status_code == 500
        assert r.json()["detail"] == "Pipeline not loaded"


def test_drift_ok_batch(client):
    payload = {
        "customers": [
            {
                "tenure": 1,
                "MonthlyCharges": 10.0,
                "TotalCharges": "10.0",
                "Contract": "Month-to-month",
                "PaymentMethod": "Electronic check",
            },
            {
                "tenure": 2,
                "MonthlyCharges": 20.0,
                "TotalCharges": "40.0",
                "Contract": "Two year",
                "PaymentMethod": "Mailed check",
            },
        ]
    }
    r = client.post("/drift", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "drift" in data
    assert "data_drift" in data["drift"]
    assert "prediction_drift" in data["drift"]