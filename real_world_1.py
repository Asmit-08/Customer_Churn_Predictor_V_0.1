
import pandas as pd

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Gaussian Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True, random_state=42)
}

# 1. Load and prepare the dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    encoding= "latin-1"
)


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

X = df.drop(columns=["Churn"])
y = df["Churn"]

# 2. Define feature columns
categorical_columns = [
    "gender", "Partner", "Dependents", "PhoneService",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "MultipleLines"
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

for model_name, model in models.items():

    if model_name == "SVM":

        model_pipeline = Pipeline(steps=[
            ("Preprocessor", pre),
            ("Scale", StandardScaler()),
            ("Classifier", model)
        ])
    else:
        model_pipeline = Pipeline(steps=[
                    ("Preprocessor", pre),
                    ("Classifier", model)
                ])

    # Train Gaussian Naive Bayes using 5-fold cross-validation
    param_grids = {
        "Logistic Regression": {
            "Classifier__C": [0.1, 1, 10],
        },
        "Decision Tree": {
            "Classifier__max_depth": [None, 5, 10],
            "Classifier__min_samples_split": [2, 5, 10],
        },
        "Random Forest": {
            "Classifier__n_estimators": [100, 200],
            "Classifier__max_depth": [None, 10, 20],
        },
        "Gradient Boosting": {
            "Classifier__n_estimators": [100, 200],
            "Classifier__learning_rate": [0.01, 0.1, 0.2],
            "Classifier__max_depth": [2, 3, 5],
        },
        "Gaussian Naive Bayes": {
            "Classifier__var_smoothing": [1e-11, 1e-9, 1e-7],
        },
        "SVM": {
            "Classifier__C": [0.1, 1, 10],
            "Classifier__kernel": ["linear", "rbf"],
        },
    }

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grids[model_name],
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(Xtrain, ytrain)

    print(f"\n {model_name} — TRAINING RESULTS")
    print("Best parameters:", grid_search.best_params_)
    print("Best CV ROC-AUC:", grid_search.best_score_)

    best_model = grid_search.best_estimator_


    val_proba = best_model.predict_proba(Xval)[:, 1]

    # Use the default probability threshold of 0.50 initially
    val_pred = (val_proba >= 0.50).astype(int)

    print(f"\n {model_name} — VALIDATION RESULTS")
    print("Accuracy:", accuracy_score(yval, val_pred))
    print("Precision:", precision_score(yval, val_pred, zero_division=0))
    print("Recall:", recall_score(yval, val_pred, zero_division=0))
    print("F1:", f1_score(yval, val_pred, zero_division=0))
    print("ROC-AUC:", roc_auc_score(yval, val_proba))


# 6. Tune GaussianNB's variance smoothing
# param_grid = {
#     "Classifier__var_smoothing": [
#         1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6
#     ]
# }

# grid_search = GridSearchCV(
#     estimator=model_pipeline,
#     param_grid=param_grid,
#     scoring="roc_auc",
#     cv=5,
#     n_jobs=-1,
#     verbose=1,
#     return_train_score=True
# )

# grid_search.fit(Xtrain, ytrain)

# print("Best parameters:", grid_search.best_params_)
# print("Best CV ROC-AUC:", grid_search.best_score_)

# best_model = grid_search.best_estimator_

# # 7. Select threshold using validation data only
# val_proba = best_model.predict_proba(Xval)[:, 1]

# thresholds = [0.20, 0.30, 0.40, 0.50, 0.65]
# threshold_results = []

# print("\nVALIDATION RESULTS")

# for threshold in thresholds:
#     val_pred = (val_proba >= threshold).astype(int)

#     precision = precision_score(
#         yval, val_pred, zero_division=0
#     )
#     recall = recall_score(
#         yval, val_pred, zero_division=0
#     )
#     f1 = f1_score(
#         yval, val_pred, zero_division=0
#     )

#     threshold_results.append({
#         "threshold": threshold,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1
#     })

#     print(
#         f"Threshold: {threshold:.2f} | "
#         f"Precision: {precision:.3f} | "
#         f"Recall: {recall:.3f} | "
#         f"F1: {f1:.3f}"
#     )

# best_result = max(
#     threshold_results,
#     key=lambda result: result["f1"]
# )

# best_threshold = best_result["threshold"]

# print("\nSelected threshold:", best_threshold)
# print("Best validation F1:", best_result["f1"])

# # 8. Final test evaluation — performed only once
# test_proba = best_model.predict_proba(Xtest)[:, 1]
# test_pred = (test_proba >= best_threshold).astype(int)

# print("\nFINAL TEST RESULTS")
# print("Threshold:", best_threshold)
# print("Accuracy:", accuracy_score(ytest, test_pred))
# print(
#     "Precision:",
#     precision_score(ytest, test_pred, zero_division=0)
# )
# print(
#     "Recall:",
#     recall_score(ytest, test_pred, zero_division=0)
# )
# print("F1:", f1_score(ytest, test_pred, zero_division=0))
# print("ROC-AUC:", roc_auc_score(ytest, test_proba))

# print("\nTest classification report:")
# print(classification_report(ytest, test_pred, zero_division=0))

# print("Test confusion matrix:")
# print(confusion_matrix(ytest, test_pred))
