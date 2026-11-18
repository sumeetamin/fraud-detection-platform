import os
import uuid

from fastapi.testclient import TestClient

from fraud.api import app
from fraud.core import FEATURES


def test_production_postgres_serves_reloaded_active_model():
    assert os.environ["FRAUD_ENV"] == "production"
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["storage"] == "ready"
    version = health.json()["model_version"]

    features = {key: 0.0 for key in FEATURES}
    features["amount"] = 25.0
    tx_id = "CI-" + uuid.uuid4().hex
    headers = {"X-API-Key": os.environ["FRAUD_API_TOKEN"]}
    scored = client.post("/predict", json={"transaction_id": tx_id, "features": features}, headers=headers)
    assert scored.status_code == 200
    assert scored.json()["model_version"] == version
    assert client.post("/labels", json={"transaction_id": tx_id, "is_fraud": 1}, headers=headers).status_code == 200
    assert client.get("/queue", headers=headers).status_code == 200
