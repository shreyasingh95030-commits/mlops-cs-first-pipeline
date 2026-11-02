import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from app.config import (
    MODEL_ALIAS,
    MODEL_DIR,
    MODEL_NAME,
    configure_tracking,
)

MIN_TEST_ACCURACY = 0.90
N_ESTIMATORS = 100
RANDOM_STATE = 42


def train() -> None:
    configure_tracking()
    mlflow.set_experiment("iris-classification")

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=iris.target,
    )

    model_parameters = {
        "model_type": "RandomForestClassifier",
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
    }

    with mlflow.start_run() as run:
        mlflow.log_params(model_parameters)
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, average="macro", zero_division=0
        )

        if accuracy < MIN_TEST_ACCURACY:
            raise ValueError(
                f"Model rejected: test accuracy {accuracy:.2f} is below "
                f"the minimum threshold of {MIN_TEST_ACCURACY:.2f}"
            )

        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
        })

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "iris_model.joblib"
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="iris-model",
            registered_model_name=MODEL_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        client = MlflowClient()
        registered_versions = client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )
        current_model_version = next(
            (
                version
                for version in registered_versions
                if version.run_id == run.info.run_id
            ),
            None,
        )
        if current_model_version is None:
            raise RuntimeError(
                "The model was logged, but its registered version "
                "could not be found."
            )

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=MODEL_ALIAS,
            version=current_model_version.version,
        )

        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Test accuracy: {accuracy:.3f}")
        print(f"Test precision: {precision:.3f}")
        print(f"Test recall: {recall:.3f}")
        print(f"Model saved to {model_path}")
        print(
            f"Registered {MODEL_NAME} version "
            f"{current_model_version.version} as @{MODEL_ALIAS}"
        )


if __name__ == "__main__":
    train()
