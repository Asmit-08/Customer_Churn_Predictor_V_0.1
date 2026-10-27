from fastapi import FastAPI
from schemas import CustomerData


app = FastAPI(title="ML Production",
              description="First ML production level API")

@app.get("/health")
def get_health():
    return{
        "Status" : "Ok",
        "Description": "Your Service is working chad."
    }


# @app.post("/predict")
# def predict(data:CustomerData):
