
import joblib
import pandas as pd
from fastapi import FastAPI
from schemas import CustomerData

app = FastAPI(
    title="ML Production",
    description="First ML production level API"
)

# Load the fitted pipeline and metadata once when the app starts.
artifact = joblib.load("churn_model.joblib")
model_pipeline = artifact["pipeline"]
threshold = artifact["threshold"]


@app.get("/health")
def get_health():
    return {
        "Status": "Ok",
        "Description": "Your service is working."
    }


@app.post("/predict")
def predict(data: CustomerData):
    customer_df = pd.DataFrame([data.model_dump()])

    churn_probability = model_pipeline.predict_proba(customer_df)[0, 1]
    prediction = int(churn_probability >= threshold)

    return {
        "prediction": prediction,
        "churn_probability": float(churn_probability),
        "threshold": threshold,
        "message": (
            "Customer will churn."
            if prediction == 1
            else "Customer will not churn."
        )
    }
