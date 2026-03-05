# FraudGuard AI - Intelligent Fraud Transaction Detection System

A production-grade machine learning system for detecting fraudulent financial transactions. Built with an intelligent 13-stage ML pipeline, FastAPI backend, and React dashboard.

## Architecture

```
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

## Prerequisites

- Python 3.9+
- Node.js 18+
- npm 9+

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pvrpavan/Fraud_transaction_detection.git
cd Fraud_transaction_detection
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Dataset

Download the PaySim synthetic financial dataset from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place the CSV file in a `data/` directory:

```bash
mkdir -p data
# Place your dataset CSV file in the data/ directory
# e.g., data/paysim.csv
```

### 4. Run the ML Pipeline

```bash
python run.py --data data/paysim.csv
```

Optional arguments:
- `--config config/config.yaml` - Custom configuration file
- `--log-level DEBUG` - Set logging level (DEBUG, INFO, WARNING, ERROR)

The pipeline will:
- Load and intelligently sample data (max 1M rows from 6.3M)
- Engineer features and handle class imbalance
- Train and compare multiple ML models
- Optimize hyperparameters with cross-validation
- Build ensemble models
- Generate SHAP explanations
- Create visualizations
- Save the best model and results to `outputs/`

### 5. Start the Backend API

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

### 6. Start the Frontend Dashboard

```bash
cd frontend/fraud-dashboard
npm install
npm run dev
```

The dashboard will be available at `http://localhost:5173`.

## Configuration

Edit `config/config.yaml` to customize:

```yaml
data:
  max_rows: 1000000          # Maximum rows to use for training
  target_column: "isFraud"   # Target variable name
  chunk_size: 100000         # Chunk size for memory-efficient loading

sampling:
  strategy: "hybrid"         # stratified, fraud_focused, clustering, anomaly, hybrid
  fraud_ratio: 0.1           # Target fraud ratio after sampling

models:
  algorithms:                # Models to train
    - logistic_regression
    - random_forest
    - gradient_boosting
    - xgboost
    - lightgbm
    - catboost

imbalance:
  method: "auto"             # auto, smote, smoteenn, adasyn, class_weight
  
tuning:
  n_iter: 50                 # Number of hyperparameter combinations
  cv_folds: 5                # Cross-validation folds
```

## Performance Targets

| Metric | Research Paper Baseline | Our Target | 
|--------|----------------------|------------|
| Accuracy | ~90% | >95% |
| ROC-AUC | ~0.90 | ~0.99 |
| Fraud Recall | ~85% | >90% |

## Tech Stack

- **ML Pipeline**: Python, scikit-learn, XGBoost, LightGBM, CatBoost, SHAP
- **Backend**: FastAPI, Uvicorn
- **Frontend**: React, TypeScript, Vite, Tailwind CSS, Recharts, Lucide Icons
- **Data Processing**: Pandas, NumPy, imbalanced-learn

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

## License

MIT
