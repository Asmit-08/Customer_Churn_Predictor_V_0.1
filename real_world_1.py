
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load and prepare the dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

df.drop(
    columns=["customerID", "MultipleLines", "PaymentMethod"],
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

X = df.drop(columns=["Churn"])
y = df["Churn"]

# 2. Define feature columns
categorical_columns = [
    "gender", "Partner", "Dependents", "PhoneService",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling"
]

numerical_columns = [
    col for col in X.columns
    if col not in categorical_columns
]

# 3. Preprocessing
pre = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        ),
        (
            "numerical",
            SimpleImputer(strategy="median"),
            numerical_columns
        )
    ],
    sparse_threshold=0
)

# 4. Gaussian Naive Bayes pipeline
model_pipeline = Pipeline(steps=[
    ("Preprocessor", pre),
    ("Classifier", GaussianNB())
])

# 5. Stratified 60/20/20 train-validation-test split
Xtrainval, Xtest, ytrainval, ytest = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

Xtrain, Xval, ytrain, yval = train_test_split(
    Xtrainval, ytrainval,
    test_size=0.25,
    random_state=42,
    stratify=ytrainval
)

# 6. Tune GaussianNB's variance smoothing
param_grid = {
    "Classifier__var_smoothing": [
        1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6
    ]
}

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(Xtrain, ytrain)

print("Best parameters:", grid_search.best_params_)
print("Best CV ROC-AUC:", grid_search.best_score_)

best_model = grid_search.best_estimator_

# 7. Select threshold using validation data only
val_proba = best_model.predict_proba(Xval)[:, 1]

thresholds = [0.20, 0.30, 0.40, 0.50, 0.65]
threshold_results = []

print("\nVALIDATION RESULTS")

for threshold in thresholds:
    val_pred = (val_proba >= threshold).astype(int)

    precision = precision_score(
        yval, val_pred, zero_division=0
    )
    recall = recall_score(
        yval, val_pred, zero_division=0
    )
    f1 = f1_score(
        yval, val_pred, zero_division=0
    )

    threshold_results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )

best_result = max(
    threshold_results,
    key=lambda result: result["f1"]
)

best_threshold = best_result["threshold"]

print("\nSelected threshold:", best_threshold)
print("Best validation F1:", best_result["f1"])

# 8. Final test evaluation — performed only once
test_proba = best_model.predict_proba(Xtest)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

print("\nFINAL TEST RESULTS")
print("Threshold:", best_threshold)
print("Accuracy:", accuracy_score(ytest, test_pred))
print(
    "Precision:",
    precision_score(ytest, test_pred, zero_division=0)
)
print(
    "Recall:",
    recall_score(ytest, test_pred, zero_division=0)
)
print("F1:", f1_score(ytest, test_pred, zero_division=0))
print("ROC-AUC:", roc_auc_score(ytest, test_proba))

print("\nTest classification report:")
print(classification_report(ytest, test_pred, zero_division=0))

print("Test confusion matrix:")
print(confusion_matrix(ytest, test_pred))
