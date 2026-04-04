"""
FraudGuard - Fraud Transaction Detection API

Production-grade FastAPI backend serving the fraud detection model results,
predictions, and dashboard data. Built as a final year college project.
"""

import json
import os
import pickle
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# State
# ============================================================

_model_data = None
_pipeline_results = None
_startup_time = None


def _load_model():
    """Load the trained model from disk."""
    global _model_data
    model_path = os.path.join(PROJECT_ROOT, "outputs", "models", "best_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            _model_data = pickle.load(f)
        return True
    return False


def _load_results():
    """Load pipeline results from disk."""
    global _pipeline_results
    results_path = os.path.join(PROJECT_ROOT, "outputs", "reports", "pipeline_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            _pipeline_results = json.load(f)
        return True
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model and results on startup."""
    global _startup_time
    _startup_time = datetime.now(timezone.utc)
    _load_model()
    _load_results()
    yield


app = FastAPI(
    title="FraudGuard - Fraud Transaction Detection API",
    description=(
        "Production-grade ML-powered fraud detection system using an ensemble of "
        "6 machine learning models with SHAP explainability. Built as a final year "
        "college project demonstrating real-world fraud detection capabilities."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Data Models
# ============================================================

class TransactionInput(BaseModel):
    """Single transaction for fraud prediction."""
    step: int = Field(default=1, description="Time step of the transaction (1 hour per step)")
    type: str = Field(default="TRANSFER", description="Transaction type: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN")
    amount: float = Field(default=10000.0, description="Transaction amount")
    oldbalanceOrg: float = Field(default=50000.0, description="Origin account balance before transaction")
    newbalanceOrig: float = Field(default=40000.0, description="Origin account balance after transaction")
    oldbalanceDest: float = Field(default=0.0, description="Destination account balance before transaction")
    newbalanceDest: float = Field(default=10000.0, description="Destination account balance after transaction")

    class Config:
        json_schema_extra = {
            "example": {
                "step": 1,
                "type": "TRANSFER",
                "amount": 181000.0,
                "oldbalanceOrg": 181000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            }
        }


class BatchPredictionInput(BaseModel):
    """Batch of transactions for prediction."""
    transactions: list[dict]


class PredictionResponse(BaseModel):
    """Prediction result for a single transaction."""
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    confidence: float
    risk_factors: list[str] = []
    recommendation: str = ""


def _analyze_risk_factors(transaction: TransactionInput) -> list[str]:
    """Analyze transaction characteristics to identify risk factors."""
    factors = []
    if transaction.type in ("TRANSFER", "CASH_OUT"):
        factors.append(f"High-risk transaction type: {transaction.type}")
    if transaction.amount > 200000:
        factors.append(f"Very large transaction amount: {transaction.amount:,.2f}")
    elif transaction.amount > 50000:
        factors.append(f"Large transaction amount: {transaction.amount:,.2f}")
    if transaction.oldbalanceOrg > 0 and transaction.newbalanceOrig == 0:
        factors.append("Complete account drain: entire balance transferred out")
    balance_change = transaction.oldbalanceOrg - transaction.newbalanceOrig
    if transaction.amount > 0 and abs(balance_change - transaction.amount) > 0.01:
        factors.append("Balance change does not match transaction amount")
    if transaction.oldbalanceDest == 0 and transaction.newbalanceDest == 0 and transaction.type == "TRANSFER":
        factors.append("Destination balance unchanged after receiving transfer")
    if transaction.amount > transaction.oldbalanceOrg and transaction.oldbalanceOrg > 0:
        factors.append("Transaction amount exceeds available balance")
    return factors


def _get_recommendation(is_fraud: bool, probability: float, risk_level: str) -> str:
    """Generate a recommendation based on prediction results."""
    if is_fraud and risk_level == "HIGH":
        return "BLOCK immediately. Flag for manual review by fraud investigation team."
    elif is_fraud and risk_level == "MEDIUM":
        return "HOLD transaction. Request additional verification from account holder."
    elif probability > 0.3:
        return "MONITOR closely. Elevated risk detected - apply enhanced due diligence."
    else:
        return "APPROVE transaction. Risk within acceptable parameters."


# ============================================================
# API Routes
# ============================================================

@app.get("/", tags=["System"])
async def root():
    """API health check and system overview."""
    return {
        "name": "FraudGuard - Fraud Transaction Detection API",
        "version": "2.0.0",
        "status": "running",
        "uptime": str(datetime.now(timezone.utc) - _startup_time) if _startup_time else "unknown",
        "model_loaded": _model_data is not None,
        "results_available": _pipeline_results is not None,
        "endpoints": {
            "docs": "/docs",
            "health": "/api/health",
            "predict": "/api/predict",
            "results": "/api/results/summary",
            "plots": "/api/plots",
        },
    }


@app.get("/api/health", tags=["System"])
async def health_check():
    """Detailed health check for monitoring."""
    model_loaded = _model_data is not None or _load_model()
    results_loaded = _pipeline_results is not None or _load_results()
    return {
        "status": "healthy" if model_loaded and results_loaded else "degraded",
        "model_status": "loaded" if model_loaded else "not_loaded",
        "results_status": "available" if results_loaded else "not_available",
        "model_name": _model_data.get("model_name") if _model_data else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/status")
async def get_status():
    """Get system status and available data."""
    model_loaded = _model_data is not None or _load_model()
    results_loaded = _pipeline_results is not None or _load_results()

    plots_dir = os.path.join(PROJECT_ROOT, "outputs", "plots")
    available_plots = []
    if os.path.exists(plots_dir):
        available_plots = [
            f for f in os.listdir(plots_dir)
            if f.endswith((".png", ".jpg", ".svg"))
        ]

    return {
        "model_loaded": model_loaded,
        "results_available": results_loaded,
        "available_plots": available_plots,
        "model_name": _model_data.get("model_name") if _model_data else None,
    }


@app.get("/api/results", tags=["Results"])
async def get_results():
    """Get complete pipeline results."""
    if _pipeline_results is None:
        if not _load_results():
            raise HTTPException(
                status_code=404,
                detail="No pipeline results found. Run the ML pipeline first: python run.py --data data/filtered_rows.csv",
            )
    return _pipeline_results


@app.get("/api/results/summary", tags=["Results"])
async def get_results_summary():
    """Get a comprehensive summary of the pipeline results for the dashboard."""
    if _pipeline_results is None:
        if not _load_results():
            raise HTTPException(
                status_code=404,
                detail="No pipeline results found. Run the ML pipeline first.",
            )

    final_eval = _pipeline_results.get("final_evaluation", {})
    dataset_info = _pipeline_results.get("dataset_info", {})
    sampling = _pipeline_results.get("sampling", {})
    model_comparison = _pipeline_results.get("model_comparison", [])
    imbalance = _pipeline_results.get("imbalance", {})
    tuning = _pipeline_results.get("hyperparameter_tuning", {})
    preprocessing = _pipeline_results.get("preprocessing", {})

    return {
        "dataset": {
            "total_transactions": dataset_info.get("total_rows", 0),
            "fraud_transactions": dataset_info.get("fraud_count", 0),
            "legitimate_transactions": dataset_info.get("legitimate_count", 0),
            "fraud_ratio": dataset_info.get("fraud_ratio", 0),
        },
        "sampling": sampling,
        "preprocessing": {
            "n_features": preprocessing.get("n_features", 0),
            "feature_names": preprocessing.get("feature_names", []),
        },
        "imbalance_handling": imbalance,
        "model_comparison": model_comparison,
        "best_model": {
            "name": final_eval.get("model_name", ""),
            "accuracy": final_eval.get("accuracy", 0),
            "precision": final_eval.get("precision", 0),
            "recall": final_eval.get("recall", 0),
            "f1_score": final_eval.get("f1_score", 0),
            "roc_auc": final_eval.get("roc_auc", 0),
            "specificity": final_eval.get("specificity", 0),
            "mcc": final_eval.get("mcc", 0),
        },
        "confusion_matrix": final_eval.get("confusion_matrix", {}),
        "hyperparameter_tuning": tuning,
        "pipeline_time": _pipeline_results.get("pipeline_time", 0),
    }


@app.get("/api/results/evaluations", tags=["Results"])
async def get_all_evaluations():
    """Get detailed evaluation results for all trained models."""
    if _pipeline_results is None:
        if not _load_results():
            raise HTTPException(status_code=404, detail="No results found.")

    return _pipeline_results.get("all_evaluations", {})


@app.get("/api/results/feature-importance", tags=["Results"])
async def get_feature_importance():
    """Get SHAP-based feature importance data."""
    if _pipeline_results is None:
        if not _load_results():
            raise HTTPException(status_code=404, detail="No results found.")

    explainability = _pipeline_results.get("explainability", {})
    return {
        "feature_importance": explainability.get("feature_importance", {}),
        "top_features": explainability.get("top_features", []),
    }


@app.post("/api/predict", tags=["Prediction"], response_model=PredictionResponse)
async def predict_transaction(transaction: TransactionInput):
    """Predict if a single transaction is fraudulent with risk analysis."""
    if _model_data is None:
        if not _load_model():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Run the ML pipeline first.",
            )

    try:
        import pandas as pd

        model = _model_data["model"]
        preprocessor = _model_data["preprocessor"]
        feature_names = _model_data["feature_names"]

        # Convert to dataframe
        df = pd.DataFrame([transaction.model_dump()])

        # Preprocess
        X, _ = preprocessor.transform(df)

        # Predict
        prediction = model.predict(X)[0]
        probability = float(
            model.predict_proba(X)[0][1]
            if hasattr(model, "predict_proba")
            else prediction
        )

        is_fraud = bool(prediction)
        risk_level = "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
        confidence = float(max(probability, 1 - probability))
        risk_factors = _analyze_risk_factors(transaction)
        recommendation = _get_recommendation(is_fraud, probability, risk_level)

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            risk_factors=risk_factors,
            recommendation=recommendation,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/predict/batch", tags=["Prediction"])
async def predict_batch(batch: BatchPredictionInput):
    """Predict fraud for a batch of transactions."""
    if _model_data is None:
        if not _load_model():
            raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        import pandas as pd

        model = _model_data["model"]
        preprocessor = _model_data["preprocessor"]

        df = pd.DataFrame(batch.transactions)
        X, _ = preprocessor.transform(df)

        predictions = model.predict(X)
        probabilities = (
            model.predict_proba(X)[:, 1]
            if hasattr(model, "predict_proba")
            else predictions.astype(float)
        )

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "index": i,
                "is_fraud": bool(pred),
                "fraud_probability": float(prob),
                "risk_level": (
                    "HIGH" if prob > 0.7
                    else "MEDIUM" if prob > 0.3
                    else "LOW"
                ),
            })

        fraud_count = sum(1 for r in results if r["is_fraud"])

        return {
            "total_transactions": len(results),
            "fraud_detected": fraud_count,
            "legitimate": len(results) - fraud_count,
            "fraud_rate": fraud_count / len(results) if results else 0,
            "predictions": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/api/plots/{filename}", tags=["Visualizations"])
async def get_plot(filename: str):
    """Serve a generated plot image."""
    plots_dir = os.path.join(PROJECT_ROOT, "outputs", "plots")
    file_path = os.path.join(plots_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Plot not found: {filename}")

    return FileResponse(file_path, media_type="image/png")


@app.get("/api/plots", tags=["Visualizations"])
async def list_plots():
    """List all available visualization plots."""
    plots_dir = os.path.join(PROJECT_ROOT, "outputs", "plots")
    if not os.path.exists(plots_dir):
        return {"plots": [], "total": 0}

    plots = [
        {
            "filename": f,
            "url": f"/api/plots/{f}",
            "size": os.path.getsize(os.path.join(plots_dir, f)),
            "title": f.replace(".png", "").replace("_", " ").title(),
        }
        for f in sorted(os.listdir(plots_dir))
        if f.endswith((".png", ".jpg", ".svg"))
    ]
    return {"plots": plots, "total": len(plots)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
