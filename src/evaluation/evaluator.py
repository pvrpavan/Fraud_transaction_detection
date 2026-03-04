"""
Model evaluation module.

Comprehensive performance assessment with multiple metrics
focused on fraud detection effectiveness.
"""

import json
import logging
import os
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for fraud detection."""

    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.evaluation_results: dict = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "model",
    ) -> dict:
        """
        Comprehensive evaluation of model predictions.

        Returns dict with all metrics.
        """
        logger.info(f"Evaluating {model_name}...")

        # Core metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        mcc = matthews_corrcoef(y_true, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Specificity (true negative rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # ROC curve data
        fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_true, y_prob)

        # Precision-Recall curve data
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
            y_true, y_prob
        )

        # Classification report
        report = classification_report(
            y_true, y_pred, target_names=["Legitimate", "Fraud"], output_dict=True
        )

        results = {
            "model_name": model_name,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "avg_precision": float(avg_precision),
            "mcc": float(mcc),
            "specificity": float(specificity),
            "false_positive_rate": float(fpr),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "roc_curve": {
                "fpr": fpr_curve.tolist(),
                "tpr": tpr_curve.tolist(),
            },
            "pr_curve": {
                "precision": pr_precision.tolist(),
                "recall": pr_recall.tolist(),
            },
            "classification_report": report,
        }

        self.evaluation_results[model_name] = results

        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
        logger.info(f"  MCC:       {mcc:.4f}")
        logger.info(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        return results

    def compare_models(self, results: dict) -> dict:
        """Compare multiple model evaluation results."""
        comparison = {}
        for name, result in results.items():
            if "error" not in result:
                comparison[name] = {
                    "accuracy": result.get("accuracy", 0),
                    "precision": result.get("precision", 0),
                    "recall": result.get("recall", 0),
                    "f1_score": result.get("f1_score", result.get("f1", 0)),
                    "roc_auc": result.get("roc_auc", 0),
                }
        return comparison

    def save_results(self, filename: str = "evaluation_results.json") -> str:
        """Save evaluation results to JSON."""
        filepath = os.path.join(self.output_dir, filename)

        # Convert numpy arrays in results for JSON serialization
        serializable = {}
        for name, result in self.evaluation_results.items():
            serializable[name] = self._make_serializable(result)

        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")
        return filepath

    def _make_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def generate_report(self, model_name: str) -> str:
        """Generate a human-readable evaluation report."""
        if model_name not in self.evaluation_results:
            return f"No results found for {model_name}"

        r = self.evaluation_results[model_name]
        cm = r["confusion_matrix"]

        report = f"""
{'='*60}
FRAUD DETECTION MODEL EVALUATION REPORT
{'='*60}

Model: {r['model_name']}

PERFORMANCE METRICS
{'-'*40}
  Accuracy:           {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%)
  Precision:          {r['precision']:.4f}
  Recall (Fraud):     {r['recall']:.4f}
  F1-Score:           {r['f1_score']:.4f}
  ROC-AUC:            {r['roc_auc']:.4f}
  Avg Precision:      {r['avg_precision']:.4f}
  MCC:                {r['mcc']:.4f}
  Specificity:        {r['specificity']:.4f}
  False Positive Rate:{r['false_positive_rate']:.4f}

CONFUSION MATRIX
{'-'*40}
  True Negatives:  {cm['true_negatives']:>8,}
  False Positives: {cm['false_positives']:>8,}
  False Negatives: {cm['false_negatives']:>8,}
  True Positives:  {cm['true_positives']:>8,}

INTERPRETATION
{'-'*40}
  Fraud Detection Rate: {r['recall']*100:.1f}% of fraudulent transactions detected
  False Alarm Rate:     {r['false_positive_rate']*100:.2f}% of legitimate transactions flagged
  When flagging fraud:  {r['precision']*100:.1f}% are actually fraudulent

{'='*60}
"""
        # Save report
        report_path = os.path.join(
            self.output_dir, f"{model_name}_report.txt"
        )
        with open(report_path, "w") as f:
            f.write(report)

        return report
