"""
Visualization module for fraud detection results.

Generates publication-quality plots for model evaluation,
data analysis, and presentation.
"""

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FraudVisualizer:
    """Generate comprehensive visualizations for fraud detection results."""

    # Professional color palette
    COLORS = {
        "primary": "#6366F1",      # Indigo
        "secondary": "#EC4899",    # Pink
        "success": "#10B981",      # Emerald
        "danger": "#EF4444",       # Red
        "warning": "#F59E0B",      # Amber
        "info": "#3B82F6",         # Blue
        "legitimate": "#10B981",   # Green
        "fraud": "#EF4444",        # Red
        "background": "#1E1B4B",   # Dark indigo
        "text": "#E2E8F0",         # Light gray
    }

    MODEL_COLORS = [
        "#6366F1", "#EC4899", "#10B981", "#F59E0B",
        "#3B82F6", "#8B5CF6", "#14B8A6", "#F97316",
    ]

    def __init__(self, output_dir: str = "outputs/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._setup_style()

    def _setup_style(self):
        """Set up matplotlib style for professional plots."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "figure.facecolor": "#0F172A",
            "axes.facecolor": "#1E293B",
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#E2E8F0",
            "text.color": "#E2E8F0",
            "xtick.color": "#94A3B8",
            "ytick.color": "#94A3B8",
            "grid.color": "#334155",
            "grid.alpha": 0.5,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "figure.dpi": 150,
        })

    def plot_class_distribution(
        self, y: np.ndarray, title: str = "Transaction Distribution"
    ) -> str:
        """Plot fraud vs legitimate transaction distribution."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Count plot
        unique, counts = np.unique(y, return_counts=True)
        labels = ["Legitimate", "Fraud"]
        colors = [self.COLORS["legitimate"], self.COLORS["fraud"]]

        bars = axes[0].bar(labels, counts, color=colors, edgecolor="white", linewidth=0.5)
        axes[0].set_title("Transaction Count", fontweight="bold")
        axes[0].set_ylabel("Count")
        for bar, count in zip(bars, counts):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.02,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontweight="bold",
                color="#E2E8F0",
            )

        # Pie chart
        axes[1].pie(
            counts,
            labels=labels,
            colors=colors,
            autopct="%1.2f%%",
            startangle=90,
            explode=(0, 0.1),
            textprops={"color": "#E2E8F0", "fontweight": "bold"},
            wedgeprops={"edgecolor": "white", "linewidth": 0.5},
        )
        axes[1].set_title("Class Distribution", fontweight="bold")

        plt.suptitle(title, fontsize=16, fontweight="bold", color="#E2E8F0")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "class_distribution.png")
        plt.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def plot_confusion_matrix(
        self, cm_dict: dict, model_name: str = "Model"
    ) -> str:
        """Plot confusion matrix as a heatmap."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        cm = np.array([
            [cm_dict["true_negatives"], cm_dict["false_positives"]],
            [cm_dict["false_negatives"], cm_dict["true_positives"]],
        ])

        fig, ax = plt.subplots(figsize=(8, 7))

        # Custom colormap
        cmap = LinearSegmentedColormap.from_list(
            "custom", ["#1E293B", "#6366F1", "#EC4899"]
        )

        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.colorbar(im, ax=ax, shrink=0.8)

        labels = ["Legitimate", "Fraud"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_yticklabels(labels, fontsize=12)
        ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=13, fontweight="bold")

        # Add text annotations
        for i in range(2):
            for j in range(2):
                text_color = "white" if cm[i, j] > cm.max() / 2 else "#E2E8F0"
                ax.text(
                    j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color=text_color,
                )

        ax.set_title(
            f"Confusion Matrix - {model_name}",
            fontsize=15, fontweight="bold", pad=15,
        )
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"confusion_matrix_{model_name}.png")
        plt.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def plot_roc_curves(self, model_results: dict) -> str:
        """Plot ROC curves for all models."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))

        for i, (name, result) in enumerate(model_results.items()):
            if "roc_curve" in result:
                fpr = result["roc_curve"]["fpr"]
                tpr = result["roc_curve"]["tpr"]
                auc_val = result["roc_auc"]
                color = self.MODEL_COLORS[i % len(self.MODEL_COLORS)]
                ax.plot(
                    fpr, tpr,
                    color=color,
                    linewidth=2.5,
                    label=f"{name} (AUC={auc_val:.4f})",
                )

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], "w--", alpha=0.3, linewidth=1)

        ax.set_xlabel("False Positive Rate", fontsize=13, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=13, fontweight="bold")
        ax.set_title(
            "ROC Curves - Model Comparison",
            fontsize=15, fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=10, framealpha=0.8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "roc_curves.png")
        plt.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def plot_precision_recall_curves(self, model_results: dict) -> str:
        """Plot precision-recall curves for all models."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))

        for i, (name, result) in enumerate(model_results.items()):
            if "pr_curve" in result:
                precision = result["pr_curve"]["precision"]
                recall = result["pr_curve"]["recall"]
                color = self.MODEL_COLORS[i % len(self.MODEL_COLORS)]
                ax.plot(
                    recall, precision,
                    color=color,
                    linewidth=2.5,
                    label=f"{name} (AP={result.get('avg_precision', 0):.4f})",
                )

        ax.set_xlabel("Recall", fontsize=13, fontweight="bold")
        ax.set_ylabel("Precision", fontsize=13, fontweight="bold")
        ax.set_title(
            "Precision-Recall Curves",
            fontsize=15, fontweight="bold",
        )
        ax.legend(loc="lower left", fontsize=10, framealpha=0.8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "precision_recall_curves.png")
        plt.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def plot_feature_importance(
        self, importance: dict, title: str = "Feature Importance", top_n: int = 15
    ) -> str:
        """Plot feature importance as horizontal bar chart."""
        import matplotlib.pyplot as plt

        # Sort and take top N
        sorted_imp = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )

        features = list(reversed(list(sorted_imp.keys())))
        values = list(reversed(list(sorted_imp.values())))

        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(features)))
        bars = ax.barh(features, values, color=colors, edgecolor="white", linewidth=0.3)

        ax.set_xlabel("Importance Score", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=9,
                color="#94A3B8",
            )

        plt.tight_layout()
        path = os.path.join(self.output_dir, "feature_importance.png")
        plt.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def plot_model_comparison(self, comparison_data: list) -> str:
        """Plot model comparison bar chart."""
        import matplotlib.pyplot as plt

        if not comparison_data:
            logger.warning("No comparison data to plot")
            return ""

        models = [d["model"] for d in comparison_data]
        metrics = ["precision", "recall", "f1", "roc_auc"]
        metric_labels = ["Precision", "Recall", "F1-Score", "ROC-AUC"]

        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(models))
        width = 0.2

        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [d.get(metric, 0) for d in comparison_data]
            color = self.MODEL_COLORS[i % len(self.MODEL_COLORS)]
            bars = ax.bar(
                x + i * width, values, width,
                label=label, color=color,
                edgecolor="white", linewidth=0.3,
            )

        ax.set_xlabel("Model", fontsize=13, fontweight="bold")
        ax.set_ylabel("Score", fontsize=13, fontweight="bold")
        ax.set_title(
            "Model Performance Comparison",
            fontsize=15, fontweight="bold",
        )
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
        ax.legend(fontsize=10, framealpha=0.8)
        ax.set_ylim([0, 1.1])
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "model_comparison.png")
        plt.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def plot_amount_distribution(self, df, target_col: str = "isFraud") -> str:
        """Plot transaction amount distribution for fraud vs legitimate."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        if "amount" in df.columns:
            fraud = df[df[target_col] == 1]["amount"]
            legit = df[df[target_col] == 0]["amount"]

            # Histogram
            axes[0].hist(
                legit, bins=50, alpha=0.7, color=self.COLORS["legitimate"],
                label="Legitimate", density=True,
            )
            axes[0].hist(
                fraud, bins=50, alpha=0.7, color=self.COLORS["fraud"],
                label="Fraud", density=True,
            )
            axes[0].set_xlabel("Amount", fontweight="bold")
            axes[0].set_ylabel("Density", fontweight="bold")
            axes[0].set_title("Amount Distribution", fontweight="bold")
            axes[0].legend()
            axes[0].set_xlim(0, fraud.quantile(0.99))

            # Log-scale box plot
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = [
                    np.log1p(legit.values),
                    np.log1p(fraud.values),
                ]
            bp = axes[1].boxplot(
                data,
                labels=["Legitimate", "Fraud"],
                patch_artist=True,
            )
            bp["boxes"][0].set_facecolor(self.COLORS["legitimate"])
            bp["boxes"][1].set_facecolor(self.COLORS["fraud"])
            for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
                for item in bp[element]:
                    item.set_color("#E2E8F0")
            axes[1].set_ylabel("Log(Amount + 1)", fontweight="bold")
            axes[1].set_title("Amount Box Plot (Log Scale)", fontweight="bold")

        plt.suptitle(
            "Transaction Amount Analysis",
            fontsize=16, fontweight="bold", color="#E2E8F0",
        )
        plt.tight_layout()
        path = os.path.join(self.output_dir, "amount_distribution.png")
        plt.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def generate_all_plots(
        self,
        y_true: np.ndarray,
        evaluation_results: dict,
        feature_importance: Optional[dict],
        comparison_data: list,
        df=None,
    ) -> dict:
        """Generate all visualization plots."""
        logger.info("Generating all visualization plots...")
        plots = {}

        # Class distribution
        try:
            plots["class_distribution"] = self.plot_class_distribution(y_true)
        except Exception as e:
            logger.warning(f"Failed to plot class distribution: {e}")

        # Confusion matrices
        for name, result in evaluation_results.items():
            if "confusion_matrix" in result:
                try:
                    plots[f"confusion_matrix_{name}"] = self.plot_confusion_matrix(
                        result["confusion_matrix"], name
                    )
                except Exception as e:
                    logger.warning(f"Failed to plot confusion matrix for {name}: {e}")

        # ROC curves
        try:
            plots["roc_curves"] = self.plot_roc_curves(evaluation_results)
        except Exception as e:
            logger.warning(f"Failed to plot ROC curves: {e}")

        # Precision-Recall curves
        try:
            plots["pr_curves"] = self.plot_precision_recall_curves(evaluation_results)
        except Exception as e:
            logger.warning(f"Failed to plot PR curves: {e}")

        # Feature importance
        if feature_importance:
            try:
                plots["feature_importance"] = self.plot_feature_importance(
                    feature_importance
                )
            except Exception as e:
                logger.warning(f"Failed to plot feature importance: {e}")

        # Model comparison
        if comparison_data:
            try:
                plots["model_comparison"] = self.plot_model_comparison(comparison_data)
            except Exception as e:
                logger.warning(f"Failed to plot model comparison: {e}")

        # Amount distribution
        if df is not None and "amount" in df.columns:
            try:
                plots["amount_distribution"] = self.plot_amount_distribution(df)
            except Exception as e:
                logger.warning(f"Failed to plot amount distribution: {e}")

        logger.info(f"Generated {len(plots)} plots")
        return plots
