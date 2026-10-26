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

result = []

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

    best_model = grid_search.best_estimator_


    val_proba = best_model.predict_proba(Xval)[:, 1]

    i = 0
    while i<1:     
        val_pred = (val_proba >= i).astype(int)
        res = {
            "model_name": model_name,
            "best_params": grid_search.best_params_,
            "threshold": i,
            "cv_roc_auc": grid_search.best_score_,
            "val_accuracy": accuracy_score(yval, val_pred),
            "val_precision": precision_score(yval, val_pred, zero_division=0),
            "val_recall": recall_score(yval, val_pred, zero_division=0),
            "val_f1": f1_score(yval, val_pred, zero_division=0),
            "val_roc_auc": roc_auc_score(yval, val_proba)
        }
        result.append(res)
        i += 0.1


results_df = pd.DataFrame(result)
results_df = results_df.sort_values(by="val_roc_auc", ascending=False)
print(results_df.to_string())
