from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_reports_registered_model():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_name"] == "iris-classifier"
    assert response.json()["model_version"]


def test_predict_returns_iris_class():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"prediction": 0, "class_name": "setosa"}


def test_predict_rejects_non_positive_measurements():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 0,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 422
