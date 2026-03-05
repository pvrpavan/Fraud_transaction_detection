# FraudGuard AI - Intelligent Fraud Transaction Detection System

A production-grade machine learning system for detecting fraudulent financial transactions. Built with an intelligent 13-stage ML pipeline, FastAPI backend, and React dashboard.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Step-by-Step Setup Guide](#step-by-step-setup-guide)
5. [Running the ML Pipeline](#running-the-ml-pipeline)
6. [Running the Backend API](#running-the-backend-api)
7. [Running the Frontend Dashboard](#running-the-frontend-dashboard)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)
10. [API Usage Examples](#api-usage-examples)
11. [Performance Targets](#performance-targets)
12. [Tech Stack](#tech-stack)
13. [License](#license)

---

## Architecture

```text
Fraud_transaction_detection/
├── src/                          # ML Pipeline
│   ├── data/
│   │   ├── data_loader.py        # Memory-efficient chunked data loading
│   │   ├── data_sampler.py       # 5 intelligent sampling strategies
│   │   └── data_preprocessor.py  # Feature engineering & preprocessing
│   ├── models/
│   │   ├── model_trainer.py      # Multi-model training (6+ algorithms)
│   │   ├── hyperparameter_tuner.py # RandomizedSearchCV optimization
│   │   └── ensemble_builder.py   # Voting & stacking ensembles
│   ├── evaluation/
│   │   ├── evaluator.py          # Comprehensive metrics & evaluation
│   │   └── explainer.py          # SHAP-based explainability
│   ├── visualization/
│   │   └── visualizer.py         # Publication-quality visualizations
│   └── pipeline.py               # Main 13-stage pipeline orchestrator
├── backend/
│   └── app/
│       └── main.py               # FastAPI REST API
├── frontend/
│   └── fraud-dashboard/          # React + TypeScript dashboard
├── config/
│   └── config.yaml               # Central configuration
├── run.py                        # CLI entry point
├── requirements.txt              # Python dependencies
└── outputs/                      # Generated after pipeline run
    ├── models/                   # Saved trained models
    ├── plots/                    # Generated visualizations
    └── reports/                  # Evaluation reports
```

## Features

### ML Pipeline (13 Stages)
1. **Data Loading** - Memory-efficient chunked loading for large datasets (6.3M+ rows)
2. **Intelligent Sampling** - 5 strategies: stratified, fraud-focused, clustering-based, anomaly-focused, hybrid
3. **Feature Engineering** - Automatic feature creation, encoding, scaling, and selection
4. **Class Imbalance Handling** - Auto-selects best technique: SMOTE, SMOTEENN, ADASYN, class weighting
5. **Multi-Model Training** - Trains 6+ algorithms: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
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

Before starting, make sure you have the following installed on your machine:

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

Creating a virtual environment keeps this project's dependencies isolated from your system Python.

**On Linux / macOS:**

```bash
python3 -m venv venv
```

**On Windows (Command Prompt):**

```cmd
python -m venv venv
```

**On Windows (PowerShell):**

```powershell
python -m venv venv
```

This creates a `venv/` folder inside your project directory.

### 3. Activate the Virtual Environment

You **must** activate the environment every time you open a new terminal to work on this project.

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

After activation, your terminal prompt will show `(venv)` at the beginning, e.g.:

```text
(venv) user@machine:~/Fraud_transaction_detection$
```

### 4. Install Python Dependencies

With the virtual environment activated, install all required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs the following major libraries:

- **numpy, pandas** - Data manipulation
- **scikit-learn** - Machine learning algorithms
- **xgboost, lightgbm, catboost** - Gradient boosting models
- **imbalanced-learn** - SMOTE and other resampling techniques
- **shap** - Model explainability
- **matplotlib, seaborn** - Visualization
- **fastapi, uvicorn** - Backend API server
- **pyyaml** - Configuration file parsing

> **Tip:** If you encounter installation errors with `lightgbm` or `catboost`, try:
>
> ```bash
> pip install lightgbm --no-binary :all:
> pip install catboost
> ```

### 5. Download the Dataset

The pipeline uses the **PaySim** synthetic financial dataset (~6.3 million transactions, ~470 MB CSV).

1. Go to: https://www.kaggle.com/datasets/ealaxi/paysim1
2. Click **Download** (you need a free Kaggle account)
3. Extract the ZIP file
4. Place the CSV file in the `data/` directory:

```bash
mkdir -p data
# Move or copy the downloaded CSV into the data/ folder
# The file is typically named: PS_20174392719_1491204167307_log.csv
cp ~/Downloads/PS_20174392719_1491204167307_log.csv data/paysim.csv
```

> **Dataset Info:**
>
> - **Rows:** ~6,362,620 transactions
> - **Fraud transactions:** ~8,213 (0.13% of total)
> - **Columns:** step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

---

## Running the ML Pipeline

### Basic Usage

```bash
python run.py --data data/paysim.csv
```

This single command runs the entire 13-stage pipeline.

### Advanced Options

```bash
# Run with custom config file
python run.py --data data/paysim.csv --config config/config.yaml

# Run with debug logging (verbose output)
python run.py --data data/paysim.csv --log-level DEBUG

# Run with default dataset path from config
python run.py
```

### What the Pipeline Does

When you run the command, the pipeline executes the following stages in order:

| Stage | Name                      | Description                                                  |
|-------|---------------------------|--------------------------------------------------------------|
| 1     | Data Loading              | Loads the CSV in chunks; preserves ALL fraud rows            |
| 2     | Intelligent Sampling      | Reduces dataset to ~1M rows while keeping all ~8,213 fraud   |
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

Training time depends on your hardware. Below are approximate times:

| Hardware                        | Estimated Time |
|---------------------------------|---------------|
| **Laptop (8 GB RAM, 4 cores)** | 15 - 30 minutes |
| **Desktop (16 GB RAM, 8 cores)** | 8 - 15 minutes |
| **Server (32+ GB RAM, 16 cores)** | 5 - 10 minutes |
| **Cloud GPU instance**         | 3 - 8 minutes  |

> **Memory note:** The pipeline loads a maximum of **1,000,000 rows** into memory (not the full 6.3M). Peak RAM usage is approximately **2-4 GB** during model training with SMOTE resampling.

### Expected Output

After the pipeline finishes, you will see a summary like:

```text
============================================================
FINAL RESULTS SUMMARY
============================================================
Best Model:  tuned_xgboost
Accuracy:    0.9995 (99.95%)
Precision:   0.9800
Recall:      0.9100
F1-Score:    0.9438
ROC-AUC:     0.9950
Pipeline Time: 420.5s
============================================================

Results saved to: outputs/reports/
Plots saved to:   outputs/plots/
Model saved to:   outputs/models/best_model.pkl
```

**Generated files:**

```text
outputs/
├── models/
│   └── best_model.pkl          # Trained model (load with pickle)
├── plots/
│   ├── roc_curves.png          # ROC curves for all models
│   ├── confusion_matrix.png    # Confusion matrix
│   ├── feature_importance.png  # Top features by SHAP
│   └── ...                     # Other visualizations
└── reports/
    ├── pipeline_results.json   # Full results in JSON format
    └── evaluation_report.txt   # Human-readable evaluation
```

### Expected Dataset After Loading

The fraud-preserving loader produces:

| Metric               | Expected Value     |
|----------------------|--------------------|
| **Total rows**       | ~1,000,000         |
| **Fraud rows**       | ~8,213 (all kept)  |
| **Legitimate rows**  | ~991,787           |
| **Fraud ratio**      | ~0.82%             |

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
cd frontend/fraud-dashboard
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
  max_training_rows: 1000000     # Max rows to use (fraud-preserving sampling)
  start_row: 1000000             # Starting row offset for flexible windowing
  target_fraud_ratio: 0.5        # Dynamic balancing (0.0 = disabled, 0.5 = 50/50)
  chunk_size: 100000             # Chunk size for memory-efficient CSV reading
  test_size: 0.2                 # Train/test split ratio
  random_state: 42               # Random seed for reproducibility

sampling:
  strategy: "intelligent"        # stratified | fraud_focused | clustering
                                 # anomaly_focused | hybrid | intelligent

imbalance:
  method: "auto"                 # auto | smote | smoteenn | adasyn | class_weight
  sampling_strategy: 0.3         # SMOTE target minority ratio

models:
  candidates:                    # Models to train and compare
    - logistic_regression
    - random_forest
    - gradient_boosting
    - xgboost
    - lightgbm
    - catboost

tuning:
  method: "random"               # grid | random | bayesian
  n_iter: 50                     # Number of hyperparameter combinations to try
  cv_folds: 5                    # Cross-validation folds
```

### Flexible Row Windowing

You can train on different sections of the dataset by changing `start_row`:

```yaml
# Train on rows 0 to 1,000,000
data:
  start_row: 0
  max_training_rows: 1000000

# Train on rows 1,000,000 to 2,000,000
data:
  start_row: 1000000
  max_training_rows: 1000000
```

> **Note:** Fraud transactions are **always** collected from the entire file regardless of `start_row`. Only legitimate transactions are windowed.

### Dynamic Balanced Sampling

Use `target_fraud_ratio` to automatically balance the fraud vs legitimate split. The loader dynamically adjusts the number of legitimate rows so that the final dataset has the desired fraud ratio.

```yaml
# 50/50 balanced dataset (~8,213 fraud + ~8,213 legit = ~16,426 total)
data:
  target_fraud_ratio: 0.5

# 10% fraud (~8,213 fraud + ~73,917 legit = ~82,130 total)
data:
  target_fraud_ratio: 0.1

# Disabled (default) - fill up to max_rows with legit
data:
  target_fraud_ratio: 0.0
```

| `target_fraud_ratio` | Fraud rows | Legit rows  | Total rows  |
|----------------------|-----------|-------------|-------------|
| 0.0 (default)        | ~8,213    | ~991,787    | ~1,000,000  |
| 0.1                  | ~8,213    | ~73,917     | ~82,130     |
| 0.3                  | ~8,213    | ~19,164     | ~27,377     |
| 0.5                  | ~8,213    | ~8,213      | ~16,426     |

> **Tip:** A `target_fraud_ratio` of `0.5` gives equal fraud/legit data which can improve model recall. Use `0.0` for the largest possible training set.

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'xgboost'` | Make sure your venv is activated and run `pip install -r requirements.txt` |
| `FileNotFoundError: Dataset not found` | Download the PaySim dataset and place it in `data/` |
| `MemoryError` during training | Reduce `max_training_rows` in `config/config.yaml` (try 500000) |
| `SMOTE error with n_jobs` | This is fixed in the current version. If you see this, pull the latest code |
| Pipeline hangs at SHAP | SHAP can be slow; reduce `shap_sample_size` in config (try 500) |
| `ImportError: libgomp` (Linux) | Install OpenMP: `sudo apt-get install libgomp1` |

### Deactivating the Virtual Environment

When you are done working, deactivate the venv:

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

### Get Pipeline Results Summary

```bash
curl http://localhost:8000/api/results/summary
```

### Get Feature Importance

```bash
curl http://localhost:8000/api/results/feature-importance
```

---

## Performance Targets

| Metric | Research Paper Baseline | Our Target |
|--------|----------------------|------------|
| Accuracy | ~90% | >95% |
| ROC-AUC | ~0.90 | ~0.99 |
| Fraud Recall | ~85% | >90% |

---

## Tech Stack

- **ML Pipeline**: Python, scikit-learn, XGBoost, LightGBM, CatBoost, SHAP
- **Backend**: FastAPI, Uvicorn
- **Frontend**: React, TypeScript, Vite, Tailwind CSS, Recharts, Lucide Icons
- **Data Processing**: Pandas, NumPy, imbalanced-learn

---

## License

MIT
