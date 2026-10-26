
import joblib
from real_world_1 import train_and_evaluate

# Train and evaluate the model
model_pipeline, metrics = train_and_evaluate()

# Store the fitted pipeline and metadata
artifact = {
    "pipeline": model_pipeline,
    "threshold": 0.40,
    "model_name": "RandomForestClassifier",
    "model_params": {
        "max_depth": 10,
        "n_estimators": 200,
        "random_state": 42
    },
    **metrics
}

joblib.dump(artifact, "churn_model.joblib")

print("Model artifact saved successfully.")
print("Metrics:", metrics)
