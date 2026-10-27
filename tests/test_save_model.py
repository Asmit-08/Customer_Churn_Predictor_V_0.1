
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "churn_model.joblib"
DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def test_artifact_loads():
    artifact = joblib.load(MODEL_PATH)

    assert isinstance(artifact, dict)


def test_metadata_valid():
    artifact = joblib.load(MODEL_PATH)

    required_keys = {
        "pipeline",
        "threshold",
        "model_name",
        "model_params",
    }

    assert required_keys.issubset(artifact.keys())
    assert artifact["pipeline"] is not None
    assert 0 <= artifact["threshold"] <= 1
    assert artifact["model_name"] == "RandomForestClassifier"
    assert isinstance(artifact["model_params"], dict)


def test_prediction_probabilities_valid():
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]

    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    X = df.drop(
        columns=["customerID", "PaymentMethod", "Churn"]
    ).head(5)

    probabilities = pipeline.predict_proba(X)

    assert probabilities.shape == (5, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

    # Each row contains probabilities for both classes.
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_threshold_produces_binary_predictions():
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]

    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    X = df.drop(
        columns=["customerID", "PaymentMethod", "Churn"]
    ).head(5)

    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    assert predictions.shape == (5,)
    assert set(np.unique(predictions)).issubset({0, 1})
