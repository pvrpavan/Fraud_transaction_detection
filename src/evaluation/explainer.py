"""
Model explainability module using SHAP.

Provides feature importance analysis and explanation of
individual predictions for fraud detection models.
"""

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ModelExplainer:
    """SHAP-based model explainability for fraud detection."""

    def __init__(self, output_dir: str = "outputs/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.shap_values = None
        self.feature_importance: Optional[dict] = None

    def explain_model(
        self,
        model,
        X: np.ndarray,
        feature_names: list,
        sample_size: int = 1000,
        model_name: str = "model",
    ) -> dict:
        """
        Generate SHAP explanations for the model.

        Returns dict with feature importance and SHAP values.
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed. Skipping explainability analysis.")
            return self._fallback_importance(model, feature_names)

        logger.info(f"Generating SHAP explanations for {model_name}...")

        # Sample data for SHAP (memory efficient)
        if X.shape[0] > sample_size:
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        try:
            # Use TreeExplainer for tree-based models
            if hasattr(model, "feature_importances_"):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)

                # Handle binary classification SHAP output
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Fraud class
            else:
                # Use KernelExplainer for other models
                background = shap.kmeans(X_sample, min(10, len(X_sample)))
                explainer = shap.KernelExplainer(
                    model.predict_proba
                    if hasattr(model, "predict_proba")
                    else model.predict,
                    background,
                )
                shap_values = explainer.shap_values(
                    X_sample[:min(100, len(X_sample))]
                )
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

            self.shap_values = shap_values

            # Compute feature importance from SHAP values
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance_dict = {}
            for i, name in enumerate(feature_names):
                if i < len(mean_abs_shap):
                    importance_dict[name] = float(mean_abs_shap[i])

            # Sort by importance
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )

            self.feature_importance = importance_dict

            # Generate SHAP plots
            self._generate_shap_plots(
                shap_values, X_sample, feature_names, model_name
            )

            logger.info(f"Top 5 important features:")
            for i, (feat, imp) in enumerate(list(importance_dict.items())[:5]):
                logger.info(f"  {i+1}. {feat}: {imp:.4f}")

            return {
                "feature_importance": importance_dict,
                "shap_values_shape": list(shap_values.shape),
                "top_features": list(importance_dict.keys())[:10],
            }

        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}. Using fallback importance.")
            return self._fallback_importance(model, feature_names)

    def _fallback_importance(self, model, feature_names: list) -> dict:
        """Fallback feature importance when SHAP is not available."""
        importance_dict = {}

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for i, name in enumerate(feature_names):
                if i < len(importances):
                    importance_dict[name] = float(importances[i])
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            for i, name in enumerate(feature_names):
                if i < len(coefs):
                    importance_dict[name] = float(coefs[i])

        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        self.feature_importance = importance_dict

        return {
            "feature_importance": importance_dict,
            "method": "model_native",
            "top_features": list(importance_dict.keys())[:10],
        }

    def _generate_shap_plots(
        self,
        shap_values: np.ndarray,
        X_sample: np.ndarray,
        feature_names: list,
        model_name: str,
    ) -> None:
        """Generate and save SHAP visualization plots."""
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Summary plot (bar)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                plot_type="bar",
                show=False,
                max_display=15,
            )
            plt.title(f"SHAP Feature Importance - {model_name}", fontsize=14)
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, f"shap_importance_{model_name}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

            # Summary plot (dot)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                show=False,
                max_display=15,
            )
            plt.title(f"SHAP Summary - {model_name}", fontsize=14)
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, f"shap_summary_{model_name}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

            logger.info("SHAP plots saved successfully")

        except Exception as e:
            logger.warning(f"Failed to generate SHAP plots: {e}")

    def get_feature_importance(self) -> Optional[dict]:
        """Return computed feature importance."""
        return self.feature_importance
