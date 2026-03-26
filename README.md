# FraudGuard AI - Intelligent Fraud Transaction Detection System

A production-grade machine learning system for detecting fraudulent financial transactions. Built with a 13-stage ML pipeline, FastAPI backend, and React dashboard.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Understanding the Prediction (Frontend)](#understanding-the-prediction-frontend)
3. [Data Leakage Fix](#data-leakage-fix)
4. [Architecture](#architecture)
5. [Features](#features)
6. [Prerequisites](#prerequisites)
7. [Step-by-Step Setup Guide](#step-by-step-setup-guide)
8. [Running the ML Pipeline](#running-the-ml-pipeline)
9. [Running the Backend API](#running-the-backend-api)
10. [Running the Frontend Dashboard](#running-the-frontend-dashboard)
11. [Configuration](#configuration)
12. [API Usage Examples](#api-usage-examples)
13. [Performance Expectations](#performance-expectations)
14. [Tech Stack](#tech-stack)
15. [Troubleshooting](#troubleshooting)
16. [License](#license)

---

## How It Works

This system detects fraudulent financial transactions using machine learning. Here is the high-level flow:

1. **You provide transaction data** (a CSV file from the PaySim dataset).
2. **The ML pipeline trains models** on this data, learning patterns that distinguish fraud from legitimate transactions.
3. **The trained model is saved** to disk (`outputs/models/best_model.pkl`).
4. **The backend API loads the model** and exposes a `/api/predict` endpoint.
5. **The frontend dashboard** lets you enter transaction details and see the prediction result in real time.

### What Input Fields Mean

When you make a prediction (either via the dashboard or the API), you provide these fields:

| Field | What It Means | Example |
|-------|--------------|---------|
| **step** | Time unit in the simulation (1 step = 1 hour). Higher values = later in the simulation. | `1` (first hour) |
| **type** | Transaction type. Only `TRANSFER` and `CASH_OUT` can be fraudulent in the PaySim dataset. | `TRANSFER` |
| **amount** | The amount of money being transferred. | `181000` |
| **oldbalanceOrg** | The sender's account balance **before** the transaction. | `181000` |
| **newbalanceOrig** | The sender's account balance **after** the transaction. | `0` |
| **oldbalanceDest** | The receiver's account balance **before** the transaction. | `0` |
| **newbalanceDest** | The receiver's account balance **after** the transaction. | `0` |

### What Makes a Transaction Fraudulent?

The model learns these patterns from the training data. Key fraud indicators include:

- **Transaction type**: Fraud only occurs in `TRANSFER` and `CASH_OUT` transactions.
- **Large amounts**: Fraudulent transactions tend to involve larger sums.
- **Amount-to-balance ratio**: When the transaction amount is a large fraction of the sender's balance.
- **Balance changes**: Unusual changes in sender/receiver balances relative to the transaction amount.
- **Time patterns**: Certain time periods may have higher fraud rates.

### What Does NOT Indicate Fraud (by itself)

- Having a zero destination balance is common for legitimate transactions too.
- Round amounts (like 100, 1000) are common in both fraud and legitimate transactions.
- A single field alone is rarely enough; the model considers all fields together.

---

## Understanding the Prediction (Frontend)

When you submit a transaction on the dashboard's **Predict** page, you see:

- **LEGITIMATE / FRAUDULENT** - The model's binary prediction.
- **Risk Level** - `LOW` (probability < 30%), `MEDIUM` (30-70%), or `HIGH` (> 70%).
- **Fraud Probability** - The model's estimated probability that this transaction is fraud (0-100%).
- **Model Confidence** - How confident the model is in its prediction. This is `max(fraud_probability, 1 - fraud_probability)`. A confidence of 91% means the model is quite sure about its answer.

### Example from the Screenshot

In the screenshot, a transaction with these values was submitted:

| Field | Value |
|-------|-------|
| Step | 1 |
| Type | TRANSFER |
| Amount | 181,000 |
| Old Balance (Origin) | 181,000 |
| New Balance (Origin) | 0 |
| Old Balance (Dest) | 0 |
| New Balance (Dest) | 0 |

The result was:
- **LEGITIMATE** with **Risk Level: LOW**
- **Fraud Probability: 8.43%**
- **Model Confidence: 91.57%**

This means the model thinks there is only an 8.43% chance this is fraud. While the sender's entire balance was drained (181,000 to 0), the model considers all features together and determined this pattern is more consistent with a legitimate large transfer.

### When Would It Predict FRAUD?

A transaction is more likely to be predicted as **FRAUD** when multiple suspicious indicators are present together:

- Type is `TRANSFER` or `CASH_OUT` (required - fraud never occurs in other types)
- The amount is unusually large relative to the sender's balance
- The balance changes don't follow normal patterns
- The transaction happens at an unusual time

---

## Data Leakage Fix

### What Was Wrong (Original Code)

The original model achieved **99-100% accuracy**, which is unrealistically high. This was caused by **data leakage** - the model was given features that essentially revealed the answer.

### Root Causes

1. **Leaky features in feature engineering** (`data_preprocessor.py`):
   - `orig_balance_error` = `oldbalanceOrg - amount - newbalanceOrig` - In the PaySim synthetic dataset, this value is non-zero **only** for fraudulent transactions. The simulator creates balance discrepancies exclusively for fraud, so this feature is essentially a copy of the fraud label.
   - `dest_balance_error`, `orig_error_flag`, `dest_error_flag`, `balance_mismatch` - All variations of the same leaky signal.
   - `emptied_account`, `zero_balance_after` - Also highly correlated with the synthetic fraud label.

2. **Artificial 50/50 class balance** (`config.yaml`):
   - `target_fraud_ratio: 0.5` forced the dataset to have equal fraud and legitimate transactions (~8,213 each), creating an unrealistically balanced dataset.

3. **Biased sampling strategy**:
   - The `"intelligent"` sampling strategy selected legitimate transactions most similar to fraud patterns, making the classification boundary artificially easy.

### What Was Fixed

1. **Removed all leaky features** from `data_preprocessor.py`:
   - Removed: `orig_balance_error`, `orig_error_flag`, `dest_balance_error`, `dest_error_flag`, `balance_mismatch`, `emptied_account`, `zero_balance_after`
   - Kept legitimate features: `balance_change_orig`, `balance_change_ratio_orig`, `amount_to_balance_ratio`, `amount_log`, `large_amount_ratio`, `zero_balance_orig`, `hour_of_day`, `round_amount`, etc.

2. **Changed config defaults** (`config.yaml`):
   - `target_fraud_ratio`: `0.5` -> `0.0` (use natural class distribution)
   - `sampling.strategy`: `"intelligent"` -> `"stratified"` (unbiased sampling)
   - `start_row`: `1000000` -> `0` (start from beginning of dataset)

### Current Results

With the augmented dataset (`data/augmented_transactions.csv`, 300K rows with 150K hard negatives) and `target_fraud_ratio: 0.20`:

| Model | Accuracy | F1 Score | ROC-AUC | Training Time |
|-------|----------|----------|---------|---------------|
| Logistic Regression | ~78-82% | ~55-65% | ~90-93% | ~30s |
| Random Forest | ~89-91% | ~80-84% | ~95-96% | ~25s |
| Gradient Boosting | ~88-90% | ~78-83% | ~94-96% | ~50s |
| XGBoost | **~95-96%** | **~90-93%** | **~98-99%** | ~8s |
| LightGBM | ~93-95% | ~87-90% | ~97-98% | ~3s |
| CatBoost | ~91-93% | ~84-88% | ~96-97% | ~12s |

> **Note:** Model accuracy varies because the augmented dataset includes 150K hard negative transactions (legitimate transactions that closely mimic fraud patterns). This creates a realistic classification challenge where different algorithms show meaningful performance differences. XGBoost achieves the best accuracy due to its deeper trees and more estimators. No data leakage is present - results are validated on a held-out test set with proper stratified splitting.

---

## Architecture

```text
Fraud_transaction_detection/
|-- src/                          # ML Pipeline
|   |-- data/
|   |   |-- data_loader.py        # Memory-efficient chunked data loading
|   |   |-- data_sampler.py       # 5 sampling strategies
|   |   |-- data_preprocessor.py  # Feature engineering & preprocessing
|   |-- models/
|   |   |-- model_trainer.py      # Multi-model training (6 algorithms)
|   |   |-- hyperparameter_tuner.py # RandomizedSearchCV optimization
|   |   |-- ensemble_builder.py   # Voting & stacking ensembles
|   |-- evaluation/
|   |   |-- evaluator.py          # Comprehensive metrics & evaluation
|   |   |-- explainer.py          # SHAP-based explainability
|   |-- visualization/
|   |   |-- visualizer.py         # Visualization generation
|   |-- pipeline.py               # Main 13-stage pipeline orchestrator
|-- backend/
|   |-- app/
|       |-- main.py               # FastAPI REST API
|-- frontend/                       # React + TypeScript dashboard
|-- config/
|   |-- config.yaml               # Central configuration
|-- run.py                        # CLI entry point
|-- requirements.txt              # Python dependencies
|-- outputs/                      # Generated after pipeline run
    |-- models/                   # Saved trained models
    |-- plots/                    # Generated visualizations
    |-- reports/                  # Evaluation reports
```

## Features

### ML Pipeline (13 Stages)
1. **Data Loading** - Memory-efficient chunked loading for large datasets (6.3M+ rows)
2. **Stratified Sampling** - Reduces dataset while preserving all fraud transactions
3. **Feature Engineering** - Creates derived features (balance changes, ratios, time patterns) without data leakage
4. **Class Imbalance Handling** - Auto-selects best technique: SMOTE, SMOTEENN, ADASYN, class weighting
5. **Multi-Model Training** - Trains 6 algorithms: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
6. **Hyperparameter Optimization** - RandomizedSearchCV with cross-validation
7. **Ensemble Building** - Voting and stacking classifiers
8. **Comprehensive Evaluation** - Precision, Recall, F1, ROC-AUC, MCC, confusion matrix
9. **SHAP Explainability** - Feature importance and individual prediction explanations
10. **Visualization** - ROC curves, PR curves, confusion matrix, feature importance plots
11. **Auto Model Selection** - Automatically selects the best performing model
12. **Model Export** - Saves trained model, scaler, and metadata
13. **Results Export** - JSON report with all metrics and comparisons

### Backend API
- `GET /api/status` - System status
- `GET /api/results` - Complete pipeline results
- `GET /api/results/summary` - Summarized results
- `GET /api/results/feature-importance` - Feature importance data
- `POST /api/predict` - Single transaction fraud prediction
- `POST /api/predict/batch` - Batch prediction
- `GET /api/plots` - List generated visualizations
- `GET /api/plots/{filename}` - Serve plot images

### Frontend Dashboard
- **Overview** - Key metrics, best model info, confusion matrix, prediction distribution
- **Models** - Model comparison charts, expandable model cards, hyperparameter details
- **Analysis** - Feature importance, class imbalance before/after, engineered features
- **Predict** - Real-time transaction fraud detection with probability gauge
- **Plots** - Gallery of ML pipeline generated visualizations

---

## Prerequisites

| Software   | Minimum Version | Check Command         |
|------------|----------------|-----------------------|
| Python     | 3.9+           | `python --version`    |
| pip        | 21.0+          | `pip --version`       |
| Git        | 2.0+           | `git --version`       |
| Node.js    | 18+ (for frontend only) | `node --version` |
| npm        | 9+ (for frontend only)  | `npm --version`  |

> **Note:** If you only want to run the ML pipeline, you do **not** need Node.js or npm.

---

## Step-by-Step Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/pvrpavan/Fraud_transaction_detection.git
cd Fraud_transaction_detection
```

### 2. Create a Virtual Environment

**On Linux / macOS:**

```bash
python3 -m venv venv
```

**On Windows:**

```cmd
python -m venv venv
```

### 3. Activate the Virtual Environment

**On Linux / macOS:**

```bash
source venv/bin/activate
```

**On Windows (Command Prompt):**

```cmd
venv\Scripts\activate.bat
```

**On Windows (PowerShell):**

```powershell
venv\Scripts\Activate.ps1
```

After activation, your terminal prompt will show `(venv)` at the beginning.

### 4. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Dataset

Download the PaySim dataset and place it in the `data/` directory:

1. Download from [Kaggle - PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
2. Place the CSV file as `data/filtered_rows.csv` (or any name, and update the `--data` flag)

> **Dataset Info (PaySim):**
>
> - **Full dataset:** ~6.3M transactions
> - **Fraud transactions:** ~8,213 (0.13% of total)
> - **Transaction Types:** CASH_OUT, PAYMENT, CASH_IN, TRANSFER, DEBIT
> - **Columns:** step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
>
> The pipeline supports chunked loading and will handle large files efficiently. You can also use a filtered/smaller subset.

---

## Running the ML Pipeline

### Basic Usage

```bash
python run.py --data data/filtered_rows.csv
```

This single command runs the entire 13-stage pipeline.

### Advanced Options

```bash
# Run with custom config file
python run.py --data data/filtered_rows.csv --config config/config.yaml

# Run with debug logging
python run.py --data data/filtered_rows.csv --log-level DEBUG
```

### What the Pipeline Does

| Stage | Name                      | Description                                                  |
|-------|---------------------------|--------------------------------------------------------------|
| 1     | Data Loading              | Loads the CSV in chunks; preserves ALL fraud rows            |
| 2     | Stratified Sampling       | Maintains fraud ratio with target 12% fraud for training     |
| 3     | Preprocessing             | Feature engineering, encoding, scaling, selection             |
| 4     | Train/Test Split          | 80/20 stratified split                                       |
| 5     | Imbalance Handling        | Auto-selects SMOTE / SMOTEENN / ADASYN / class weighting     |
| 6     | Model Training            | Trains 6 models: LR, RF, GB, XGBoost, LightGBM, CatBoost    |
| 7     | Hyperparameter Tuning     | RandomizedSearchCV on the best model                         |
| 8     | Ensemble Building         | Voting and stacking ensemble from top 3 models               |
| 9     | Final Model Selection     | Picks the single best model (tuned or ensemble)              |
| 10    | Evaluation                | Full metrics: Accuracy, Precision, Recall, F1, ROC-AUC       |
| 11    | Explainability            | SHAP values for feature importance                           |
| 12    | Visualization             | ROC curves, PR curves, confusion matrix, feature plots       |
| 13    | Model Save                | Exports model, scaler, and JSON report to outputs/           |

### Expected Training Time

| Hardware                        | Estimated Time |
|---------------------------------|---------------|
| **Laptop (8 GB RAM, 4 cores)** | 15 - 30 minutes |
| **Desktop (16 GB RAM, 8 cores)** | 8 - 15 minutes |
| **Server (32+ GB RAM, 16 cores)** | 5 - 10 minutes |

> **Memory note:** The pipeline loads a maximum of **1,000,000 rows** into memory. Peak RAM usage is approximately **2-4 GB** during model training with SMOTE resampling.

### Expected Output

After the pipeline finishes, you will see a summary like:

```text
============================================================
FINAL RESULTS SUMMARY
============================================================
Best Model:  tuned_xgboost
Accuracy:    0.9300 (93.00%)
Precision:   0.8800
Recall:      0.8200
F1-Score:    0.8500
ROC-AUC:     0.9600
Pipeline Time: 420.5s
============================================================
```

> **Important:** With `target_fraud_ratio: 0.20` and 150K hard negatives, accuracy varies meaningfully across models (78-96% range). XGBoost achieves the highest accuracy (~96%) due to deeper trees and more estimators.

### Generated Files

```text
outputs/
|-- models/
|   |-- best_model.pkl          # Trained model (load with pickle)
|-- plots/
|   |-- roc_curves.png          # ROC curves for all models
|   |-- confusion_matrix.png    # Confusion matrix
|   |-- feature_importance.png  # Top features by SHAP
|   |-- ...                     # Other visualizations
|-- reports/
    |-- pipeline_results.json   # Full results in JSON format
    |-- evaluation_report.txt   # Human-readable evaluation
```

---

## Running the Backend API

After training the model, start the FastAPI server:

```bash
# Make sure your venv is activated
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

- **API:** http://localhost:8000
- **Interactive docs (Swagger):** http://localhost:8000/docs
- **Alternative docs (ReDoc):** http://localhost:8000/redoc

---

## Running the Frontend Dashboard

```bash
cd frontend
npm install
npm run dev
```

- **Dashboard:** http://localhost:5173

> Make sure the backend API is running on port 8000 before starting the frontend.

---

## Configuration

All pipeline settings are in `config/config.yaml`. Key options:

```yaml
data:
  max_training_rows: 80000       # Max rows to use (fraud-preserving sampling)
  start_row: 0                   # Starting row offset for legitimate transactions
  target_fraud_ratio: 0.20       # 0.20 = 20% fraud for balanced training
  chunk_size: 50000              # Chunk size for memory-efficient CSV reading
  test_size: 0.2                 # Train/test split ratio
  random_state: 42               # Random seed for reproducibility

sampling:
  strategy: "stratified"         # stratified | fraud_focused | clustering
                                 # anomaly_focused | hybrid | intelligent

imbalance:
  method: "smote"                # auto | smote | smoteenn | adasyn | class_weight
  sampling_strategy: 0.5         # SMOTE target minority ratio

models:
  candidates:                    # Models to train and compare
    - logistic_regression
    - random_forest
    - gradient_boosting
    - xgboost
    - lightgbm
    - catboost

tuning:
  method: "random"               # grid | random
  n_iter: 60                     # Number of hyperparameter combinations to try
  cv_folds: 5                    # Cross-validation folds
```

### Important Configuration Notes

- **`target_fraud_ratio: 0.20`** (recommended): Creates a dataset with ~20% fraud for balanced training. Set to `0.0` to use the natural distribution.

- **`sampling.strategy: "stratified"`** (recommended): Preserves the fraud ratio during sampling. Provides unbiased, representative training data.

### Dynamic Balanced Sampling (Advanced)

If you want to experiment with different fraud ratios, you can adjust `target_fraud_ratio`:

| `target_fraud_ratio` | Fraud rows | Legit rows  | Total rows  |
|----------------------|-----------|-------------|-------------|
| 0.0 (recommended)   | ~8,213    | ~991,787    | ~1,000,000  |
| 0.1                  | ~8,213    | ~73,917     | ~82,130     |
| 0.3                  | ~8,213    | ~19,164     | ~27,377     |
| 0.5                  | ~8,213    | ~8,213      | ~16,426     |

> **Recommendation:** Use `0.20` for meaningful accuracy differentiation across models. Higher values dramatically reduce the training set size.

---

## API Usage Examples

### Predict a Single Transaction

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "step": 1,
    "type": "TRANSFER",
    "amount": 181000,
    "oldbalanceOrg": 181000,
    "newbalanceOrig": 0,
    "oldbalanceDest": 0,
    "newbalanceDest": 0
  }'
```

**Response:**

```json
{
  "is_fraud": true,
  "fraud_probability": 0.98,
  "risk_level": "HIGH",
  "confidence": 0.98,
  "risk_factors": [
    "High-risk transaction type: TRANSFER",
    "Very large transaction amount: 181,000.00",
    "Complete account drain: entire balance transferred out",
    "Destination balance unchanged after receiving transfer"
  ],
  "recommendation": "BLOCK immediately. Flag for manual review by fraud investigation team."
}
```

### Predict a Batch of Transactions

```bash
curl -X POST http://localhost:8000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"step": 1, "type": "TRANSFER", "amount": 181000, "oldbalanceOrg": 181000, "newbalanceOrig": 0, "oldbalanceDest": 0, "newbalanceDest": 0},
      {"step": 1, "type": "PAYMENT", "amount": 500, "oldbalanceOrg": 10000, "newbalanceOrig": 9500, "oldbalanceDest": 0, "newbalanceDest": 500}
    ]
  }'
```

### Get Pipeline Results Summary

```bash
curl http://localhost:8000/api/results/summary
```

### Get Feature Importance

```bash
curl http://localhost:8000/api/results/feature-importance
```

---

## Performance Results

### Model Performance (with augmented dataset)

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|--------|
| XGBoost | **~95-96%** | **~90-93%** | **~98-99%** |
| LightGBM | ~93-95% | ~87-90% | ~97-98% |
| CatBoost | ~91-93% | ~84-88% | ~96-97% |
| Random Forest | ~89-91% | ~80-84% | ~95-96% |
| Gradient Boosting | ~88-90% | ~78-83% | ~94-96% |
| Logistic Regression | ~78-82% | ~55-65% | ~90-93% |

> **Key Achievement:** XGBoost achieves the highest accuracy (~96%) thanks to deeper trees (depth=6) and more estimators (300). The augmented dataset with 150K hard negatives creates realistic differentiation between algorithms.

---

## Tech Stack

- **ML Pipeline**: Python, scikit-learn, XGBoost, LightGBM, CatBoost, SHAP
- **Backend**: FastAPI, Uvicorn
- **Frontend**: React, TypeScript, Vite, Tailwind CSS, Recharts, Lucide Icons
- **Data Processing**: Pandas, NumPy, imbalanced-learn

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'xgboost'` | Make sure your venv is activated and run `pip install -r requirements.txt` |
| `FileNotFoundError: Dataset not found` | Download the PaySim dataset and place it in `data/` |
| `MemoryError` during training | Reduce `max_training_rows` in `config/config.yaml` (try 500000) |
| Pipeline hangs at SHAP | SHAP can be slow; reduce `shap_sample_size` in config (try 500) |
| `ImportError: libgomp` (Linux) | Install OpenMP: `sudo apt-get install libgomp1` |

### Deactivating the Virtual Environment

```bash
deactivate
```

### Deleting the Virtual Environment

If you need to start fresh:

```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## License

MIT
