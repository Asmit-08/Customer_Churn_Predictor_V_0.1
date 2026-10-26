import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.ensemble import RandomForestClassifier


# 1. Load dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    encoding="latin-1"
)


# 2. Basic data preparation
df.drop(
    columns=["customerID", "PaymentMethod"],
    inplace=True
)

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)


# 3. Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]


# 4. Define feature columns
categorical_columns = [
    "gender", "Partner", "Dependents", "PhoneService",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling",
    "MultipleLines"
]

numerical_columns = [
    col for col in X.columns
    if col not in categorical_columns
]


# 5. Preprocessing pipelines
categorical_pipeline = Pipeline(steps=[
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

numerical_pipeline = Pipeline(steps=[
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

pre = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        )
    ]
)


# 6. Stratified 60/20/20 train-validation-test split
Xtrainval, Xtest, ytrainval, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

Xtrain, Xval, ytrain, yval = train_test_split(
    Xtrainval,
    ytrainval,
    test_size=0.25,
    random_state=42,
    stratify=ytrainval
)


# 7. Selected Random Forest configuration
model = RandomForestClassifier(
    max_depth=10,
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model_pipeline = Pipeline(steps=[
    ("Preprocessor", pre),
    ("Classifier", model)
])


# 8. Train model
model_pipeline.fit(Xtrain, ytrain)


# 9. Validation evaluation
threshold = 0.40

val_proba = model_pipeline.predict_proba(Xval)[:, 1]
val_pred = (val_proba >= threshold).astype(int)

print("\nRANDOM FOREST — VALIDATION RESULTS")
print("Parameters:", {
    "max_depth": 10,
    "n_estimators": 200,
    "random_state": 42
})
print("Threshold:", threshold)
print("Accuracy:", accuracy_score(yval, val_pred))
print("Precision:", precision_score(
    yval, val_pred, zero_division=0
))
print("Recall:", recall_score(
    yval, val_pred, zero_division=0
))
print("F1:", f1_score(
    yval, val_pred, zero_division=0
))
print("ROC-AUC:", roc_auc_score(yval, val_proba))

print("\nValidation Confusion Matrix:")
print(confusion_matrix(yval, val_pred))

print("\nValidation Classification Report:")
print(classification_report(
    yval, val_pred, zero_division=0
))


# 10. Final held-out test evaluation
# Run only after model and threshold selection are complete.
test_proba = model_pipeline.predict_proba(Xtest)[:, 1]
test_pred = (test_proba >= threshold).astype(int)

print("\nRANDOM FOREST — FINAL TEST RESULTS")
print("Threshold:", threshold)
print("Accuracy:", accuracy_score(ytest, test_pred))
print("Precision:", precision_score(
    ytest, test_pred, zero_division=0
))
print("Recall:", recall_score(
    ytest, test_pred, zero_division=0
))
print("F1:", f1_score(
    ytest, test_pred, zero_division=0
))
print("ROC-AUC:", roc_auc_score(ytest, test_proba))

print("\nTest Confusion Matrix:")
print(confusion_matrix(ytest, test_pred))

print("\nTest Classification Report:")
print(classification_report(
    ytest, test_pred, zero_division=0
))




def train_and_evaluate():
    # Move your existing code here, starting from loading the dataset.
    # Keep the same preprocessing, split, model, and evaluation logic.

    # After fitting and evaluating:
    metrics = {
        "validation_f1": f1_score(yval, val_pred),
        "test_f1": f1_score(ytest, test_pred),
        "test_roc_auc": roc_auc_score(ytest, test_proba)
    }

    return model_pipeline, metrics


if __name__ == "__main__":
    train_and_evaluate()
