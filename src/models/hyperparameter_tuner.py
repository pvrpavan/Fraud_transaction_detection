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
            "solver": ["lbfgs"],
            "max_iter": [5000],
        },
        "random_forest": {
            "n_estimators": [80, 100, 120],
            "max_depth": [4, 5, 6, 7],
            "min_samples_split": [15, 20, 25],
            "min_samples_leaf": [8, 10, 12],
            "max_features": ["sqrt", "log2"],
        },
        "gradient_boosting": {
            "n_estimators": [60, 80, 100],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.03, 0.05],
            "subsample": [0.6, 0.7, 0.8],
            "min_samples_split": [15, 20, 25],
            "min_samples_leaf": [8, 10, 12],
        },
        "xgboost": {
            "n_estimators": [60, 80, 100],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.03, 0.05],
            "subsample": [0.6, 0.7, 0.8],
            "colsample_bytree": [0.6, 0.7, 0.8],
            "min_child_weight": [8, 10, 12],
            "gamma": [0.3, 0.5, 0.7],
            "reg_alpha": [0.5, 1.0, 2.0],
            "reg_lambda": [3, 5, 7],
        },
        "lightgbm": {
            "n_estimators": [60, 80, 100],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.03, 0.05],
            "subsample": [0.6, 0.7, 0.8],
            "colsample_bytree": [0.6, 0.7, 0.8],
            "min_child_samples": [25, 30, 40],
            "num_leaves": [10, 15, 20],
            "reg_alpha": [0.5, 1.0, 2.0],
            "reg_lambda": [3, 5, 7],
        },
        "catboost": {
            "iterations": [60, 80, 100],
            "depth": [2, 3, 4],
            "learning_rate": [0.01, 0.03, 0.05],
            "l2_leaf_reg": [7, 10, 15],
            "border_count": [32, 64, 128],
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
