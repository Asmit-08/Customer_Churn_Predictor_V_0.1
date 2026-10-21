# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to leave a service provider using customer account and service information.

## Overview

Customer churn is an important business problem because identifying customers at risk of leaving can help companies develop targeted retention strategies.

This project builds a binary classification pipeline using the Telco Customer Churn dataset. It includes data preprocessing, cross-validation, hyperparameter tuning, decision-threshold selection, and final model evaluation.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- NumPy
- Git and GitHub

## Machine Learning Workflow

1. **Data preprocessing**
   - Load and inspect the dataset.
   - Convert `TotalCharges` to numeric values.
   - Handle missing numerical values using median imputation.
   - Encode categorical features using one-hot encoding.
   - Separate input features from the target variable.

2. **Data splitting**
   - Use stratified splitting to preserve class proportions.
   - Allocate data to training, validation, and test sets.

3. **Model development**
   - Build a preprocessing and classification pipeline.
   - Train a `GradientBoostingClassifier`.
   - Use `GridSearchCV` with five-fold cross-validation to tune hyperparameters.

4. **Decision-threshold selection**
   - Evaluate candidate thresholds using validation precision, recall, and F1-score.
   - Select the threshold with the highest validation F1-score among the tested candidates.

5. **Model evaluation**
   - Evaluate the selected configuration on the test set.
   - Report accuracy, precision, recall, F1-score, ROC-AUC, classification report, and confusion matrix.

## Model Configuration

The best hyperparameters selected by cross-validation were:

| Parameter | Value |
|---|---:|
| Learning rate | 0.05 |
| Maximum tree depth | 2 |
| Minimum samples per leaf | 5 |
| Number of estimators | 150 |
| Cross-validation folds | 5 |
| Selected decision threshold | 0.30 |

The decision threshold was selected separately using validation data to optimize F1-score among the tested thresholds.

## Results

### Final test-set evaluation

| Metric | Result |
|---|---:|
| Accuracy | 76.01% |
| Precision — churn class | 53.31% |
| Recall — churn class | 77.54% |
| F1-score — churn class | 0.632 |
| ROC-AUC | 0.845 |

### Confusion matrix

Rows represent actual classes and columns represent predicted classes. Class `1` means churn, and class `0` means no churn.

| | Predicted: No churn | Predicted: Churn |
|---|---:|---:|
| Actual: No churn | 781 | 254 |
| Actual: Churn | 84 | 290 |

The model identified 290 of the 374 actual churners in the test set. However, it also flagged 254 customers who did not churn. This illustrates the trade-off between identifying more potential churners and generating false-positive predictions.

## Key Learnings

- Building reusable preprocessing and modeling pipelines.
- Encoding categorical features and imputing missing values.
- Using stratified train, validation, and test splits.
- Applying cross-validation and grid search for hyperparameter tuning.
- Understanding the trade-off between precision and recall.
- Selecting a classification threshold using validation data.
- Evaluating classification models with multiple metrics rather than accuracy alone.
- Interpreting confusion matrices in the context of a business problem.

## Limitations and Future Improvements

- Compare Gradient Boosting against a simple baseline and Logistic Regression.
- Investigate the effect of excluded features on model performance.
- Analyze feature importance and potential data leakage.
- Evaluate threshold selection using business costs and retention campaign constraints.
- Improve experiment tracking and reproducibility.
- Validate performance on an independently collected dataset before considering real-world deployment.

The current model is an experimental baseline, not a production-ready churn prediction system.

## Project Structure

```text
Churn_Predictor_V0.1/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── real_world_1.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Running the Project

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies.
4. Ensure the dataset is available at the path expected by the script.
5. Run the model.

```bash
git clone <your-repository-url>
cd Churn_Predictor_V0.1

python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and run:

```bash
pip install -r requirements.txt
python real_world_1.py
```

## Dataset

The project uses the Telco Customer Churn dataset, commonly distributed as `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

Before redistributing the dataset in a public repository, verify the original source and its applicable license or usage terms.

## Disclaimer

This project is intended for educational purposes. Its predictions should not be treated as guaranteed indicators of customer behavior.

## Author

Asmit Tiwari
