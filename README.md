# Customer Churn Predictor

A machine learning project for predicting customer churn using a trained classification model and a FastAPI-based inference API. The project covers model experimentation, hyperparameter selection, evaluation, model persistence, and API integration.

## Overview

Customer churn occurs when customers stop using a company's products or services. Predicting churn can help businesses identify customers who may be at risk and prioritize retention efforts.

This project uses the Telco Customer Churn dataset to train and evaluate classification models, select an appropriate model configuration, and expose predictions through a REST API.

### Project objectives

- Develop a machine learning model for binary customer churn classification.
- Experiment with model configurations and identify suitable hyperparameters.
- Evaluate model performance using relevant classification metrics.
- Save the selected model for reuse during inference.
- Build a FastAPI service that accepts customer data and returns predictions.

## Tech Stack

- **Language:** Python
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn
- **API framework:** FastAPI
- **Request validation:** Pydantic
- **Model serialization:** Joblib
- **API server:** Uvicorn

## Project Structure

```text
Churn_Predictor_V0.1/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── tests/
│
├── .gitignore
├── churn_model.joblib
├── main.py
├── optimum_model_testing.py
├── README.md
├── real_world_1.py
├── requirements.txt
├── save_model.py
└── schemas.py
```

### File responsibilities

| File or directory | Purpose |
|---|---|
| `data/` | Contains the dataset used for model development. |
| `tests/` | Contains project tests. |
| `real_world_1.py` | Main machine learning experimentation and model development script. |
| `optimum_model_testing.py` | Tests the selected model using the optimum hyperparameters identified during experimentation. |
| `save_model.py` | Handles saving the trained model for later inference. |
| `churn_model.joblib` | Serialized model artifact loaded by the API. |
| `main.py` | Contains the FastAPI application, including the prediction endpoint and API logic. |
| `schemas.py` | Defines the Pydantic `BaseModel` schema used to validate the POST request body for the `/predict` endpoint in `main.py`. |
| `requirements.txt` | Lists the Python dependencies required by the project. |
| `.gitignore` | Specifies files and directories that Git should ignore. |

## Dataset

The project uses the Telco Customer Churn dataset:

`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains customer demographics, account information, subscribed services, billing details, and a churn target.

- **Target variable:** `Churn`
- **Problem type:** Binary classification
- **Objective:** Predict whether a customer is likely to churn based on the available customer features.

The target variable is used during training and should not be included as an input feature when making predictions for an unseen customer.

## Model Development Workflow

The project follows a structured machine learning workflow:

1. **Data preparation:** Load the dataset and prepare the features and target variable.
2. **Model experimentation:** Train and evaluate candidate classification models.
3. **Hyperparameter selection:** Identify suitable model parameters using the experimentation workflow.
4. **Optimum model testing:** Evaluate the selected model under the chosen hyperparameters using `optimum_model_testing.py`.
5. **Model persistence:** Save the trained model using `save_model.py`.
6. **API integration:** Load the saved artifact in the FastAPI application.
7. **Inference:** Validate incoming customer data and generate predictions through `/predict`.

The training and inference workflows should use consistent preprocessing so that the model receives data in the expected format.

## Model Configuration

The current Random Forest configuration previously evaluated in the project was:

| Parameter | Value |
|---|---:|
| `n_estimators` | 200 |
| `max_depth` | 10 |
| `random_state` | 42 |
| Classification threshold | 0.40 |

These values should be confirmed against the current saved model and optimum-model testing script before being treated as the final configuration.

### Classification threshold

The classification threshold determines how the predicted churn probability is converted into a class label.

With a threshold of `0.40`, a customer is classified as a predicted churn case when the model's churn probability is greater than or equal to 0.40, assuming this is how the rule is implemented in the API.

The threshold affects the trade-off between precision and recall. It should be selected using validation results and the business cost of missed churn cases versus unnecessary retention interventions.

## Model Evaluation

The project evaluates classification performance using metrics such as:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted churn cases that are actual churn cases.
- **Recall:** Proportion of actual churn cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Breakdown of correct and incorrect predictions by class.

For churn prediction, accuracy alone may be insufficient, particularly when the dataset has imbalanced classes.

Final performance figures should be taken from the latest reproducible evaluation run. Results from earlier experiments should not be presented as current results without verification.

## FastAPI Application

The API is implemented in `main.py`, with request validation schemas defined in `schemas.py`.

### `GET /health`

A health-check endpoint for verifying that the API service responds.

A successful health check confirms that the endpoint is reachable, but does not independently prove that model inference is functioning correctly.

### `POST /predict`

The prediction endpoint accepts customer information in the request body.

The request body is validated using the Pydantic `BaseModel` schema defined in `schemas.py`. The API then uses the loaded model artifact to generate a prediction.

The exact request fields and response structure are defined by the current implementation in `schemas.py` and `main.py`.

### Interactive API documentation

When the server is running, FastAPI provides interactive documentation at:

`http://127.0.0.1:8000/docs`

The documentation can be used to inspect the available endpoints, review request schemas, and submit test requests.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Churn_Predictor_V0.1
```

Replace the placeholder with the actual repository URL.

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Verify the model artifact

Ensure that `churn_model.joblib` is present in the expected location and is compatible with the installed dependencies.

If the artifact must be regenerated, use the model-saving workflow defined in `save_model.py`.

### 5. Start the API

From the project root:

```bash
uvicorn main:app --reload
```

This command assumes that the FastAPI application instance in `main.py` is named `app`.

The API should normally be available at:

`http://127.0.0.1:8000`

## Testing and Verification

Before considering the project complete, verify the following:

- [ ] Dependencies install successfully.
- [ ] The API starts without errors.
- [ ] `GET /health` returns the expected response.
- [ ] `POST /predict` accepts valid customer data.
- [ ] Invalid input is handled appropriately.
- [ ] The saved model artifact loads successfully.
- [ ] Prediction probabilities are within the range 0 to 1, if returned.
- [ ] The configured classification threshold is applied correctly.
- [ ] The final evaluation results are reproducible.
- [ ] The test suite passes.
- [ ] The repository contains no unintended files or exposed secrets.

## Limitations

- Model performance depends on the quality and representativeness of the training data.
- Predictions are estimates and are not guarantees of future customer behavior.
- The model identifies statistical patterns but does not establish the causes of churn.
- A threshold that performs well on one dataset may not be optimal for another business.
- Offline evaluation does not establish real-world retention impact.
- Production deployment would require additional work on monitoring, data drift, access control, reliability, and retraining.

## Future Improvements

Potential future improvements include:

- Evaluating performance on new customer data.
- Monitoring prediction quality and data drift.
- Comparing models under a consistent validation strategy.
- Selecting thresholds using business costs and retention outcomes.
- Adding production-oriented monitoring, authentication, and logging.

These are potential extensions, not claims about features already implemented.

## License and Dataset Attribution

Add the appropriate software license and any required dataset attribution before public distribution.

## Author

Developed as a machine learning project focused on customer churn classification, model evaluation, and API-based inference.
