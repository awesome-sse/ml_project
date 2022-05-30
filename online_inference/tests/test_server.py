import pytest
from starlette.testclient import TestClient

from server import app


FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "id"]
DATA = [52.0, 1.0, 0.0, 118.0, 186.0, 0.0,
        2.0, 190.0, 0.0, 0.0, 1.0, 0.0, 1.0, 15.0]


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_service_reply_to_root_path(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_request(client):
    response = client.get("/predict/",
                          json={"data": [DATA], "features": FEATURES})
    assert response.status_code == 200
    assert len(response.json()) != 0


def test_predict_request_no_data(client):
    response = client.get("/predict/", json={"data": [], "features": FEATURES})
    assert response.status_code == 400
    assert "Empty input" in response.json()["detail"]
