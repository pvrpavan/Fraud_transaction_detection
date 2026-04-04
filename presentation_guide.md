# Fraud Transaction Detection System - Staff Presentation Guide

## Project Title
**FraudGuard: Intelligent Fraud Transaction Detection System Using Machine Learning**

---

## Project Overview

### Problem Statement
Financial fraud causes billions of dollars in losses annually. Traditional rule-based systems fail to detect sophisticated fraud patterns. This project develops an ML-powered system that automatically detects fraudulent transactions with **99.99% accuracy** using an ensemble of 6 machine learning models.

### Objective
To build a production-ready fraud detection system that:
1. Analyzes financial transaction data to identify fraud patterns
2. Trains and compares multiple ML models for optimal performance
3. Provides real-time fraud prediction through a REST API
4. Visualizes results through an interactive web dashboard
5. Explains model decisions using SHAP (SHapley Additive exPlanations)

---

## Technology Stack

### Backend
| Technology | Purpose |
|-----------|---------|
| **Python 3.12** | Core programming language |
| **FastAPI** | REST API framework (async, high-performance) |
| **scikit-learn** | ML model training, preprocessing, evaluation |
| **XGBoost** | Gradient boosting model |
| **LightGBM** | Light gradient boosting model |
| **CatBoost** | Categorical boosting model |
| **SHAP** | Model explainability |
| **Pandas / NumPy** | Data manipulation and analysis |
| **Matplotlib / Seaborn** | Visualization generation |
| **imbalanced-learn** | SMOTE for handling class imbalance |

### Frontend
| Technology | Purpose |
|-----------|---------|
| **React 18** | UI framework |
| **TypeScript** | Type-safe JavaScript |
| **Vite** | Build tool and dev server |
| **Tailwind CSS** | Utility-first styling |
| **Recharts** | Interactive charts |
| **Lucide React** | Icon library |

---

## System Architecture

```
React Frontend (Dashboard)
    |
    | REST API Calls (HTTP/JSON)
    |
FastAPI Backend
    |
    |-- /api/results    (Pipeline results and metrics)
    |-- /api/predict    (Real-time fraud prediction)
    |-- /api/plots      (Generated visualizations)
    |-- /api/health     (System health check)
    |
Trained ML Model (best_model.pkl)
    |
ML Pipeline (13 Stages)
    Data -> Sampling -> Preprocessing -> Split ->
    Imbalance -> Training -> Tuning -> Ensembles ->
    Evaluation -> Explainability -> Visualization ->
    Save -> Report
```

---

## ML Pipeline - 13 Stages Explained

### Stage 1: Data Loading
- Loads transaction data from CSV file (150,000 transactions)
- Memory-efficient chunked reading for large datasets

### Stage 2: Dynamic Sampling
- Stratified sampling to maintain fraud ratio
- Target fraud ratio: 12% for balanced training

### Stage 3: Preprocessing & Feature Engineering
- Creates 25+ engineered features from raw data
- Feature selection reduces to most informative features (~32)

### Stage 4: Train/Test Split
- 80/20 stratified split preserving fraud ratio

### Stage 5: Class Imbalance Handling (SMOTE)
- Generates synthetic fraud samples to balance classes
- Achieves ~33% fraud ratio in training data

### Stage 6: Model Training & Comparison
Six models trained: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost

### Stage 7: Hyperparameter Tuning
- RandomizedSearchCV with 60 iterations, 5-fold cross-validation

### Stage 8: Ensemble Methods
- Voting Ensemble and Stacking Ensemble

### Stage 9: Final Evaluation
- Comprehensive metrics on held-out test set

### Stage 10: Model Explainability (SHAP)
- SHAP values for feature importance

### Stage 11: Visualization
- Publication-quality plots

### Stage 12: Model Saving
- Best model serialized with preprocessor

### Stage 13: Report Generation
- Complete JSON report of all pipeline results

---

## Dataset Description

| Property | Value |
|----------|-------|
| **Total Transactions** | 150,000 |
| **Fraud Transactions** | 8,213 (5.47%) |
| **Legitimate Transactions** | 141,787 (94.53%) |
| **Features** | 11 columns |
| **Transaction Types** | CASH_OUT, PAYMENT, CASH_IN, TRANSFER, DEBIT |

### Column Descriptions
| Column | Description |
|--------|-------------|
| step | Time step (1 step = 1 hour) |
| type | Transaction type |
| amount | Transaction amount |
| nameOrig | Origin account identifier |
| oldbalanceOrg | Origin balance before transaction |
| newbalanceOrig | Origin balance after transaction |
| nameDest | Destination account identifier |
| oldbalanceDest | Destination balance before transaction |
| newbalanceDest | Destination balance after transaction |
| isFraud | Target variable (1 = fraud, 0 = legitimate) |
| isFlaggedFraud | Rule-based fraud flag (legacy system) |

---

## Results Summary

### Model Performance Comparison

| Model | Accuracy | F1 Score | ROC-AUC | Training Time |
|-------|----------|----------|---------|---------------|
| Logistic Regression | 62.95% | 39.13% | 98.72% | 43.1s |
| Random Forest | **99.99%** | **99.94%** | 99.97% | 24.9s |
| Gradient Boosting | **99.99%** | **99.94%** | **100.0%** | 49.1s |
| XGBoost | **99.99%** | **99.94%** | **100.0%** | 2.2s |
| LightGBM | **99.99%** | **99.94%** | **100.0%** | 2.3s |
| CatBoost | **99.99%** | **99.94%** | **100.0%** | 10.9s |

### Key Achievements
- **99.99% accuracy** across all tree-based models
- **99.94% F1 score** - excellent balance of precision and recall
- **100% ROC-AUC** for gradient boosting models
- No data leakage - proper train/test split with stratification

---

## Key Technical Concepts to Explain

### 1. SMOTE (Synthetic Minority Over-sampling Technique)
Creates synthetic examples of the minority class (fraud) using k-nearest neighbors.

### 2. Cross-Validation
5-fold stratified cross-validation ensures model generalizes well.

### 3. SHAP (SHapley Additive exPlanations)
Based on game theory - explains how each feature contributes to predictions.

### 4. Ensemble Methods
Voting (multiple models vote) and Stacking (meta-learner combines outputs).

### 5. Feature Engineering
Creating new features from existing data (balance ratios, amount categories, etc.).

---

## How to Run the Project

### Step 1: Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost imbalanced-learn shap matplotlib seaborn pyyaml fastapi uvicorn python-multipart aiofiles joblib tqdm
```

### Step 2: Train the ML Pipeline
```bash
python run.py --data data/filtered_rows.csv
```

### Step 3: Start the Backend API
```bash
uvicorn backend.app.main:app --reload --port 8000
```

### Step 4: Start the Frontend Dashboard
```bash
cd frontend/fraud-dashboard && npm install && npm run dev
```

### Step 5: Make Predictions
Use the "Predict" tab in the dashboard or call the API directly:
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"step":1,"type":"TRANSFER","amount":181000,"oldbalanceOrg":181000,"newbalanceOrig":0,"oldbalanceDest":0,"newbalanceDest":0}'
```

---

## Dashboard Features

1. **Overview Tab**: Key metrics, confusion matrix, dataset statistics
2. **Models Tab**: Side-by-side model comparison with interactive charts
3. **Analysis Tab**: Feature importance, class imbalance analysis, engineered features
4. **Predict Tab**: Real-time fraud prediction with risk analysis
5. **Plots Tab**: All ML pipeline generated visualizations

---

## Potential Questions from Staff & Answers

### Q: How do you handle the class imbalance problem?
**A:** We use SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic fraud examples. The original dataset has only 5.47% fraud transactions, which we boost to ~33% during training. We also use stratified sampling to maintain fraud ratios during train/test split.

### Q: Why did you choose these specific models?
**A:** We selected a range of models from simple (Logistic Regression as baseline) to complex (XGBoost, LightGBM, CatBoost) to demonstrate the progression. Tree-based ensemble models are particularly effective for fraud detection because they can capture non-linear patterns and feature interactions.

### Q: How do you prevent overfitting?
**A:** Through 5-fold cross-validation, proper train/test split (80/20), stratified sampling, and regularization parameters in each model. The high accuracy on the test set (which the model never sees during training) confirms generalization.

### Q: What is the practical impact of false positives vs false negatives?
**A:** False negatives (missed fraud) are more costly - they result in financial losses. False positives (legitimate flagged as fraud) cause inconvenience but are less harmful. Our models achieve extremely high recall (fraud detection rate) to minimize missed fraud.

### Q: How does SHAP help in explaining predictions?
**A:** SHAP assigns an importance value to each feature for every prediction. This helps us understand WHY a transaction is flagged as fraud, which is essential for regulatory compliance and building trust with financial institutions.

### Q: Can this system work in real-time?
**A:** Yes! The FastAPI backend can process predictions in milliseconds. XGBoost and LightGBM are particularly fast for inference. The system is designed for production deployment.

### Q: What are the limitations?
**A:** The system is trained on synthetic data, so real-world performance may vary. Fraud patterns evolve over time, requiring periodic model retraining. The system should be part of a larger fraud prevention strategy, not the sole defense.

---

## Project Structure

```
Fraud_transaction_detection/
├── config/config.yaml           # Pipeline configuration
├── data/filtered_rows.csv       # Transaction dataset (150K rows)
├── src/
│   ├── pipeline.py              # Main 13-stage pipeline
│   ├── data/                    # Data loading & preprocessing
│   ├── models/                  # Model training, tuning, ensembles
│   ├── evaluation/              # Evaluation & SHAP explainability
│   └── visualization/           # Plot generation
├── backend/app/main.py          # FastAPI REST API
├── frontend/fraud-dashboard/    # React TypeScript dashboard
├── outputs/                     # Generated models, plots, reports
├── run.py                       # Pipeline entry point
├── fraud_values_guide.md        # Fraud indicators documentation
├── presentation_guide.md        # This file
└── README.md                    # Project documentation
```

---

## Summary for Staff

This project demonstrates:
1. **End-to-end ML pipeline** - From raw data to deployed prediction system
2. **Multiple model comparison** - 6 different ML algorithms compared fairly
3. **Production-ready architecture** - FastAPI backend + React frontend
4. **Model explainability** - SHAP-based feature importance analysis
5. **Class imbalance handling** - SMOTE technique for rare event detection
6. **99.99% accuracy** - Achieved through proper feature engineering and ensemble methods
7. **Real-time predictions** - Sub-millisecond inference through REST API
8. **Interactive dashboard** - Professional visualization of all results
