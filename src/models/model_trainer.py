"""
Model training module with automatic model selection.

Trains multiple ML algorithms and compares performance to
automatically select the best model for fraud detection.
"""

import logging
import time
import warnings
from typing import Optional

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


def _get_xgboost():
    """Lazy import for XGBoost."""
    from xgboost import XGBClassifier
    return XGBClassifier


def _get_lightgbm():
    """Lazy import for LightGBM."""
    from lightgbm import LGBMClassifier
    return LGBMClassifier


def _get_catboost():
    """Lazy import for CatBoost."""
    from catboost import CatBoostClassifier
    return CatBoostClassifier


class ModelTrainer:
    """Automatic multi-model trainer and selector."""

    # Model configurations with default hyperparameters
    MODEL_CONFIGS = {
        "logistic_regression": {
            "class": LogisticRegression,
            "params": {
                "C": 1.0,
                "max_iter": 5000,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "random_state": 42,
            },
        },
        "random_forest": {
            "class": RandomForestClassifier,
            "params": {
                "n_estimators": 80,
                "max_depth": 5,
                "min_samples_split": 25,
                "min_samples_leaf": 12,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        "gradient_boosting": {
            "class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 500,
                "max_depth": 8,
                "learning_rate": 0.08,
                "subsample": 0.85,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": "sqrt",
                "random_state": 42,
            },
        },
        "xgboost": {
            "class_fn": _get_xgboost,
            "params": {
                "n_estimators": 600,
                "max_depth": 9,
                "learning_rate": 0.08,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "scale_pos_weight": 5,
                "min_child_weight": 2,
                "gamma": 0.05,
                "reg_alpha": 0.05,
                "reg_lambda": 2.0,
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "logloss",
            },
        },
        "lightgbm": {
            "class_fn": _get_lightgbm,
            "params": {
                "n_estimators": 120,
                "max_depth": 4,
                "learning_rate": 0.04,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "scale_pos_weight": 3,
                "min_child_samples": 25,
                "num_leaves": 20,
                "reg_alpha": 1.0,
                "reg_lambda": 3.0,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            },
        },
        "catboost": {
            "class_fn": _get_catboost,
            "params": {
                "iterations": 100,
                "depth": 4,
                "learning_rate": 0.04,
                "l2_leaf_reg": 7,
                "auto_class_weights": "Balanced",
                "random_state": 42,
                "verbose": 0,
            },
        },
    }

    def __init__(
        self,
        model_names: Optional[list] = None,
        cv_folds: int = 5,
        scoring: str = "f1",
    ):
        self.model_names = model_names or list(self.MODEL_CONFIGS.keys())
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.trained_models: dict = {}
        self.training_results: dict = {}
        self.best_model_name: Optional[str] = None
        self.best_model = None

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """Train all configured models and compare performance."""
        logger.info(f"Training {len(self.model_names)} models...")
        logger.info(
            f"Training set: {X_train.shape[0]:,} samples, "
            f"Validation set: {X_val.shape[0]:,} samples"
        )

        results = {}

        for name in self.model_names:
            if name not in self.MODEL_CONFIGS:
                logger.warning(f"Unknown model: {name}, skipping...")
                continue

            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Training: {name}")
                logger.info(f"{'='*50}")

                result = self._train_single_model(
                    name, X_train, y_train, X_val, y_val
                )
                results[name] = result
                self.trained_models[name] = result["model"]

                logger.info(
                    f"  {name}: F1={result['f1']:.4f}, "
                    f"ROC-AUC={result['roc_auc']:.4f}, "
                    f"Time={result['training_time']:.1f}s"
                )

            except Exception as e:
                logger.error(f"  {name}: Failed - {e}")
                results[name] = {"error": str(e)}

        self.training_results = results

        # Find best model
        valid_results = {
            k: v for k, v in results.items() if "error" not in v
        }
        if valid_results:
            self.best_model_name = max(
                valid_results, key=lambda k: valid_results[k]["f1"]
            )
            self.best_model = self.trained_models[self.best_model_name]
            logger.info(
                f"\nBest model: {self.best_model_name} "
                f"(F1={valid_results[self.best_model_name]['f1']:.4f})"
            )

        return results

    def _train_single_model(
        self,
        name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """Train a single model and evaluate it."""
        config = self.MODEL_CONFIGS[name]

        # Get model class
        if "class" in config:
            model_class = config["class"]
        else:
            model_class = config["class_fn"]()

        model = model_class(**config["params"])

        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Predict
        y_pred = model.predict(X_val)
        y_prob = (
            model.predict_proba(X_val)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_val)
            if hasattr(model, "decision_function")
            else y_pred.astype(float)
        )

        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average="binary", zero_division=0
        )
        roc_auc = roc_auc_score(y_val, y_prob)

        # Cross-validation score (suppress feature name warnings from LightGBM)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
            cv_scores = cross_val_score(
                model_class(**config["params"]),
                X_train,
                y_train,
                cv=min(self.cv_folds, 3),
                scoring=self.scoring,
                n_jobs=-1,
            )

        return {
            "model": model,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "training_time": training_time,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "report": classification_report(
                y_val, y_pred, target_names=["Legitimate", "Fraud"], output_dict=True
            ),
        }

    def get_comparison_table(self) -> list:
        """Get a comparison table of all trained models."""
        table = []
        for name, result in self.training_results.items():
            if "error" in result:
                continue
            table.append(
                {
                    "model": name,
                    "accuracy": result.get("accuracy", 0),
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1": result["f1"],
                    "roc_auc": result["roc_auc"],
                    "cv_mean": result["cv_mean"],
                    "cv_std": result["cv_std"],
                    "training_time": result["training_time"],
                    "is_best": name == self.best_model_name,
                }
            )
        return sorted(table, key=lambda x: x["f1"], reverse=True)

    def get_best_model(self) -> tuple:
        """Return the best model and its name."""
        return self.best_model, self.best_model_name
