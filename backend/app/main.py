"""
Fraud Transaction Detection API

FastAPI backend serving the fraud detection model results,
predictions, and dashboard data.
"""

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

app = FastAPI(
    title="Fraud Transaction Detection API",
    description="Production-grade ML-powered fraud detection system",
    version="1.0.0",
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
    """Single transaction for prediction."""
    step: int = 1
    type: str = "TRANSFER"
    amount: float = 10000.0
    oldbalanceOrg: float = 50000.0
    newbalanceOrig: float = 40000.0
    oldbalanceDest: float = 0.0
    newbalanceDest: float = 10000.0


class BatchPredictionInput(BaseModel):
    """Batch of transactions for prediction."""
    transactions: list[dict]


class PipelineStatus(BaseModel):
    """Pipeline execution status."""
    status: str
    message: str
    progress: Optional[float] = None


# ============================================================
# State
# ============================================================

_model_data = None
_pipeline_results = None


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


# ============================================================
# API Routes
# ============================================================

@app.get("/")
async def root():
    """API health check."""
    return {
        "name": "Fraud Transaction Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": _model_data is not None,
        "results_available": _pipeline_results is not None,
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


@app.get("/api/results")
async def get_results():
    """Get complete pipeline results."""
    if _pipeline_results is None:
        if not _load_results():
            raise HTTPException(
                status_code=404,
                detail="No pipeline results found. Run the ML pipeline first.",
            )
    return _pipeline_results


@app.get("/api/results/summary")
async def get_results_summary():
    """Get a summary of the pipeline results."""
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


@app.get("/api/results/evaluations")
async def get_all_evaluations():
    """Get evaluation results for all models."""
    if _pipeline_results is None:
        if not _load_results():
            raise HTTPException(status_code=404, detail="No results found.")

    return _pipeline_results.get("all_evaluations", {})


@app.get("/api/results/feature-importance")
async def get_feature_importance():
    """Get feature importance data."""
    if _pipeline_results is None:
        if not _load_results():
            raise HTTPException(status_code=404, detail="No results found.")

    explainability = _pipeline_results.get("explainability", {})
    return {
        "feature_importance": explainability.get("feature_importance", {}),
        "top_features": explainability.get("top_features", []),
    }


@app.post("/api/predict")
async def predict_transaction(transaction: TransactionInput):
    """Predict if a single transaction is fraudulent."""
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

        return {
            "is_fraud": bool(prediction),
            "fraud_probability": probability,
            "risk_level": (
                "HIGH" if probability > 0.7
                else "MEDIUM" if probability > 0.3
                else "LOW"
            ),
            "confidence": float(max(probability, 1 - probability)),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/predict/batch")
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
            "predictions": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/api/plots/{filename}")
async def get_plot(filename: str):
    """Serve a generated plot image."""
    plots_dir = os.path.join(PROJECT_ROOT, "outputs", "plots")
    file_path = os.path.join(plots_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Plot not found: {filename}")

    return FileResponse(file_path, media_type="image/png")


@app.get("/api/plots")
async def list_plots():
    """List all available plot files."""
    plots_dir = os.path.join(PROJECT_ROOT, "outputs", "plots")
    if not os.path.exists(plots_dir):
        return {"plots": []}

    plots = [
        {
            "filename": f,
            "url": f"/api/plots/{f}",
            "size": os.path.getsize(os.path.join(plots_dir, f)),
        }
        for f in sorted(os.listdir(plots_dir))
        if f.endswith((".png", ".jpg", ".svg"))
    ]
    return {"plots": plots}


# Try to load model and results on startup
@app.on_event("startup")
async def startup_event():
    """Load model and results on startup."""
    _load_model()
    _load_results()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
