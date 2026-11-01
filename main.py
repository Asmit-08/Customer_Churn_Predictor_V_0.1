
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, responses, status
from schemas import CustomerData
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "churn_model.joblib"

try:
    artifact = joblib.load(MODEL_PATH)

    model_pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]

except FileNotFoundError:
    logger.exception("Model artifact not found.")
    raise RuntimeError("Model artifact is missing.")

except (KeyError, TypeError):
    logger.exception("Invalid model artifact structure.")
    raise RuntimeError("Model artifact is invalid.")

except Exception:
    logger.exception("Failed to load model artifact.")
    raise RuntimeError("Model artifact could not be loaded.")


app = FastAPI(
    title="ML Production",
    description="First ML production level API"
)


@app.get("/health", status_code=status.HTTP_200_OK)
def get_health():
    return {
        "Status": "Ok",
        "Description": "Your service is working."
    }


@app.post("/predict", status_code=status.HTTP_200_OK)
def predict(data: CustomerData):
    customer_df = pd.DataFrame([data.model_dump()])

    churn_probability = model_pipeline.predict_proba(customer_df)[0, 1]
    prediction = int(churn_probability >= threshold)

    return {
        "churn_probability": float(churn_probability),
        "threshold": threshold,
        "message": (
            "Customer will churn."
            if prediction == 1
            else "Customer will not churn."
        )
    }
