import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris

from app.config import MODEL_ALIAS, MODEL_NAME, configure_tracking

configure_tracking()
client = mlflow.MlflowClient()
champion_model = client.get_model_version_by_alias(
    name=MODEL_NAME,
    alias=MODEL_ALIAS,
)
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
model_version = champion_model.version
class_names = load_iris().target_names.tolist()

app = FastAPI(title="Iris Model API")


class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/")
def root():
    return {"message": "Iris Model API is ready"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "model_version": model_version,
    }


@app.post("/predict")
def predict(data: IrisRequest):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]
    prediction = int(model.predict(features)[0])

    return {
        "prediction": prediction,
        "class_name": class_names[prediction],
    }
