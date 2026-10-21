import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

df.drop(columns=["customerID", "MultipleLines", "PaymentMethod"], inplace=True)
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)


X = df.drop(columns =["Churn"])
y = df["Churn"]

ohe = OneHotEncoder()
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

pre = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_columns),
        ("numerical", SimpleImputer(strategy="median"),
         numerical_columns)
    ]
)

model_pipeline= Pipeline(steps = [
    ("Preproccesor", pre),
    ("Scaler", StandardScaler()),
    ("Classifier", RandomForestClassifier(n_estimators=150,
                                          max_depth=5,
                                          class_weight="balanced"))
])

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.25, random_state=42)

model_pipeline.fit(Xtrain, ytrain)

thresh = 0.65

proba = model_pipeline.predict_proba(Xtest)[:, 1]
pred = (proba >= thresh).astype(int)

print("Threshold:", thresh)
print("Accuracy:", accuracy_score(ytest, pred))
print("\nConfusion Matrix:\n", confusion_matrix(ytest, pred))
print("\nClassification Report:\n",
      classification_report(ytest, pred))
print("ROC-AUC:", roc_auc_score(ytest, proba))