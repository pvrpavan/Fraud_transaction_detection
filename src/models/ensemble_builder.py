"""
Ensemble model builder module.

Creates voting classifiers, stacking models, and blended models
to improve fraud detection performance.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class EnsembleBuilder:
    """Build ensemble models from trained base models."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.ensemble_model = None
        self.ensemble_type: Optional[str] = None
        self.ensemble_results: Optional[dict] = None

    def build_voting_ensemble(
        self,
        models: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        voting: str = "soft",
    ) -> dict:
        """Build a voting classifier from the top models."""
        logger.info(f"Building {voting} voting ensemble from {len(models)} models...")

        estimators = [(name, model) for name, model in models.items()]

        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=-1,
        )

        ensemble.fit(X_train, y_train)

        y_pred = ensemble.predict(X_val)
        if voting == "soft":
            y_prob = ensemble.predict_proba(X_val)[:, 1]
        else:
            y_prob = y_pred.astype(float)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average="binary", zero_division=0
        )
        roc_auc = roc_auc_score(y_val, y_prob)

        self.ensemble_model = ensemble
        self.ensemble_type = f"voting_{voting}"

        result = {
            "model": ensemble,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "y_pred": y_pred,
            "y_prob": y_prob,
            "type": f"voting_{voting}",
            "base_models": list(models.keys()),
            "report": classification_report(
                y_val, y_pred, target_names=["Legitimate", "Fraud"], output_dict=True
            ),
        }

        self.ensemble_results = result

        logger.info(
            f"  Voting Ensemble: F1={f1:.4f}, ROC-AUC={roc_auc:.4f}"
        )

        return result

    def build_stacking_ensemble(
        self,
        models: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """Build a stacking classifier with logistic regression as meta-learner."""
        logger.info(f"Building stacking ensemble from {len(models)} models...")

        estimators = [(name, model) for name, model in models.items()]

        meta_learner = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            class_weight="balanced",
        )

        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=3,
            n_jobs=-1,
            passthrough=False,
        )

        ensemble.fit(X_train, y_train)

        y_pred = ensemble.predict(X_val)
        y_prob = ensemble.predict_proba(X_val)[:, 1]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average="binary", zero_division=0
        )
        roc_auc = roc_auc_score(y_val, y_prob)

        self.ensemble_model = ensemble
        self.ensemble_type = "stacking"

        result = {
            "model": ensemble,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "y_pred": y_pred,
            "y_prob": y_prob,
            "type": "stacking",
            "base_models": list(models.keys()),
            "meta_learner": "logistic_regression",
            "report": classification_report(
                y_val, y_pred, target_names=["Legitimate", "Fraud"], output_dict=True
            ),
        }

        self.ensemble_results = result

        logger.info(
            f"  Stacking Ensemble: F1={f1:.4f}, ROC-AUC={roc_auc:.4f}"
        )

        return result

    def build_best_ensemble(
        self,
        models: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """Try both ensemble methods and return the best one."""
        logger.info("Building and comparing ensemble methods...")

        # Select top 3 models for ensemble
        top_models = dict(list(models.items())[:3])

        results = {}

        # Try voting ensemble
        try:
            voting_result = self.build_voting_ensemble(
                top_models, X_train, y_train, X_val, y_val, voting="soft"
            )
            results["voting_soft"] = voting_result
        except Exception as e:
            logger.warning(f"Voting ensemble failed: {e}")

        # Try stacking ensemble
        try:
            stacking_result = self.build_stacking_ensemble(
                top_models, X_train, y_train, X_val, y_val
            )
            results["stacking"] = stacking_result
        except Exception as e:
            logger.warning(f"Stacking ensemble failed: {e}")

        if not results:
            logger.warning("All ensemble methods failed")
            return {}

        # Pick the best ensemble
        best_key = max(results, key=lambda k: results[k]["f1"])
        best_result = results[best_key]

        self.ensemble_model = best_result["model"]
        self.ensemble_type = best_key
        self.ensemble_results = best_result

        logger.info(
            f"Best ensemble: {best_key} "
            f"(F1={best_result['f1']:.4f}, ROC-AUC={best_result['roc_auc']:.4f})"
        )

        return best_result

    def get_ensemble_model(self):
        """Return the ensemble model."""
        return self.ensemble_model, self.ensemble_type
