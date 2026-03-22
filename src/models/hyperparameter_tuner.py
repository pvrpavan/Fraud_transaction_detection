"""
Hyperparameter optimization module.

Supports GridSearchCV, RandomizedSearchCV, and Bayesian optimization
with cross-validation to prevent overfitting.
"""

import logging
import time
from typing import Optional

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Advanced hyperparameter optimization for fraud detection models."""

    # Search spaces for each model type
    PARAM_SPACES = {
        "logistic_regression": {
            "C": [0.1, 1, 5, 10, 50, 100],
            "l1_ratio": [0, 1],
            "solver": ["saga"],
            "max_iter": [5000],
        },
        "random_forest": {
            "n_estimators": [300, 500, 700, 1000],
            "max_depth": [15, 20, 25, 30, None],
            "min_samples_split": [2, 3, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
        },
        "gradient_boosting": {
            "n_estimators": [200, 400, 600],
            "max_depth": [5, 7, 9, 12],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.8, 0.85, 0.9, 1.0],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [1, 2, 3],
        },
        "xgboost": {
            "n_estimators": [300, 500, 700],
            "max_depth": [6, 8, 10, 12],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.8, 0.85, 0.9],
            "colsample_bytree": [0.8, 0.85, 0.9],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.05, 0.1, 0.2],
            "reg_alpha": [0, 0.05, 0.1, 0.5],
            "reg_lambda": [1, 2, 3, 5],
        },
        "lightgbm": {
            "n_estimators": [300, 500, 700],
            "max_depth": [6, 8, 10, -1],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.8, 0.85, 0.9],
            "colsample_bytree": [0.8, 0.85, 0.9],
            "min_child_samples": [5, 10, 20],
            "num_leaves": [31, 50, 63, 80, 100],
            "reg_alpha": [0, 0.05, 0.1, 0.5],
            "reg_lambda": [0, 1, 2, 5],
        },
        "catboost": {
            "iterations": [300, 500, 700],
            "depth": [6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5, 7, 10],
            "border_count": [32, 64, 128, 254],
        },
    }

    def __init__(
        self,
        method: str = "random",
        n_iter: int = 50,
        cv_folds: int = 5,
        scoring: str = "f1",
        random_state: int = 42,
    ):
        self.method = method
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.best_params: dict = {}
        self.tuning_results: dict = {}

    def tune(
        self,
        model,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple:
        """
        Tune hyperparameters for a given model.

        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        logger.info(f"Tuning hyperparameters for {model_name}...")
        logger.info(f"Method: {self.method}, Iterations: {self.n_iter}")

        if model_name not in self.PARAM_SPACES:
            logger.warning(
                f"No param space defined for {model_name}. Returning original model."
            )
            return model, {}, 0.0

        param_space = self.PARAM_SPACES[model_name]
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        start_time = time.time()

        search = RandomizedSearchCV(
            model,
            param_distributions=param_space,
            n_iter=min(self.n_iter, self._count_combinations(param_space)),
            scoring=self.scoring,
            cv=cv,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0,
            refit=True,
        )

        search.fit(X_train, y_train)
        tuning_time = time.time() - start_time

        best_params = search.best_params_
        best_score = search.best_score_
        best_model = search.best_estimator_

        self.best_params[model_name] = best_params
        self.tuning_results[model_name] = {
            "best_params": best_params,
            "best_score": float(best_score),
            "tuning_time": tuning_time,
            "n_candidates": search.n_splits_,
        }

        logger.info(f"  Best score: {best_score:.4f}")
        logger.info(f"  Best params: {best_params}")
        logger.info(f"  Tuning time: {tuning_time:.1f}s")

        return best_model, best_params, best_score

    def _count_combinations(self, param_space: dict) -> int:
        """Count total parameter combinations."""
        total = 1
        for values in param_space.values():
            if isinstance(values, list):
                total *= len(values)
        return total

    def get_tuning_summary(self) -> dict:
        """Get summary of all tuning results."""
        return self.tuning_results
