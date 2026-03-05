"""
Main fraud detection pipeline orchestrator.

Coordinates all stages of the ML pipeline from data loading
through model training, evaluation, and result generation.
"""

import json
import logging
import os
import pickle
import time
from typing import Optional

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.data.data_sampler import DataSampler
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.explainer import ModelExplainer
from src.models.ensemble_builder import EnsembleBuilder
from src.models.hyperparameter_tuner import HyperparameterTuner
from src.models.model_trainer import ModelTrainer
from src.visualization.visualizer import FraudVisualizer

logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    """End-to-end fraud detection pipeline."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.results: dict = {}
        self.best_model = None
        self.best_model_name: Optional[str] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.feature_names: list = []

        # Create output directories
        for dir_key in ["model_dir", "plot_dir", "report_dir"]:
            os.makedirs(self.config["output"][dir_key], exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config

    def run(self, data_path: Optional[str] = None) -> dict:
        """
        Execute the complete fraud detection pipeline.

        Args:
            data_path: Optional override for dataset path

        Returns:
            Dictionary with all pipeline results
        """
        pipeline_start = time.time()
        logger.info("=" * 60)
        logger.info("FRAUD DETECTION PIPELINE - STARTING")
        logger.info("=" * 60)

        data_config = self.config["data"]
        file_path = data_path or data_config["raw_data_path"]

        # ============================================================
        # STAGE 1: Data Loading
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: Data Loading")
        logger.info("=" * 60)

        loader = DataLoader(
            file_path=file_path,
            chunk_size=data_config.get("chunk_size", 100000),
            max_rows=data_config.get("max_training_rows", 1_000_000),
            start_row=data_config.get("start_row", 0),
            target_fraud_ratio=data_config.get("target_fraud_ratio", 0.0),
        )

        # Get dataset info
        dataset_info = loader.get_dataset_info()
        self.results["dataset_info"] = dataset_info

        logger.info(f"Original dataset: {dataset_info['total_rows']:,} rows, "
                    f"fraud={dataset_info['fraud_count']:,}, "
                    f"legit={dataset_info['legitimate_count']:,}, "
                    f"ratio={dataset_info['fraud_ratio']:.4%}")

        # Load with fraud-priority
        df = loader.load_fraud_and_sample(
            max_rows=data_config["max_training_rows"],
        )

        # ============================================================
        # STAGE 2: Intelligent Sampling
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: Intelligent Sampling")
        logger.info("=" * 60)

        sampler = DataSampler(
            max_rows=data_config["max_training_rows"],
            random_state=data_config.get("random_state", 42),
        )

        sampling_config = self.config.get("sampling", {})
        strategy = sampling_config.get("strategy", "hybrid")

        df_sampled = sampler.sample(df, strategy=strategy)
        self.results["sampling"] = {
            "strategy": strategy,
            "original_rows": len(df),
            "sampled_rows": len(df_sampled),
            "fraud_count": int(df_sampled["isFraud"].sum()),
            "fraud_ratio": float(df_sampled["isFraud"].mean()),
        }

        logger.info(f"Strategy: {strategy} | "
                    f"Rows: {len(df):,} -> {len(df_sampled):,} | "
                    f"Fraud: {self.results['sampling']['fraud_count']:,} | "
                    f"Ratio: {self.results['sampling']['fraud_ratio']:.4%}")

        # Keep a copy for visualization
        df_for_viz = df_sampled.copy()

        # ============================================================
        # STAGE 3: Preprocessing & Feature Engineering
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: Preprocessing & Feature Engineering")
        logger.info("=" * 60)

        prep_config = self.config.get("preprocessing", {})
        self.preprocessor = DataPreprocessor(
            scaling_method=prep_config.get("scaling_method", "robust"),
            feature_selection=prep_config.get("feature_selection", True),
            variance_threshold=prep_config.get("variance_threshold", 0.01),
        )

        X, y, self.feature_names = self.preprocessor.fit_transform(
            df_sampled, target_col="isFraud"
        )

        self.results["preprocessing"] = {
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "feature_importance_mi": self.preprocessor.get_feature_importance(),
        }

        # ============================================================
        # STAGE 4: Train/Test Split
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 4: Train/Test Split")
        logger.info("=" * 60)

        test_size = data_config.get("test_size", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=data_config.get("random_state", 42),
            stratify=y,
        )

        logger.info(
            f"Train: {X_train.shape[0]:,} samples, Test: {X_test.shape[0]:,} samples"
        )
        logger.info(
            f"Train fraud ratio: {y_train.mean():.4%}, "
            f"Test fraud ratio: {y_test.mean():.4%}"
        )

        # ============================================================
        # STAGE 5: Class Imbalance Handling
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 5: Class Imbalance Handling")
        logger.info("=" * 60)

        X_train_balanced, y_train_balanced, imbalance_method = self._handle_imbalance(
            X_train, y_train
        )
        self.results["imbalance"] = {
            "method": imbalance_method,
            "original_train_size": len(y_train),
            "balanced_train_size": len(y_train_balanced),
            "original_fraud_ratio": float(y_train.mean()),
            "balanced_fraud_ratio": float(y_train_balanced.mean()),
        }

        # ============================================================
        # STAGE 6: Model Training & Comparison
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 6: Model Training & Comparison")
        logger.info("=" * 60)

        model_config = self.config.get("models", {})
        trainer = ModelTrainer(
            model_names=model_config.get("candidates", None),
            cv_folds=self.config.get("tuning", {}).get("cv_folds", 5),
            scoring=self.config.get("tuning", {}).get("scoring", "f1"),
        )

        training_results = trainer.train_all(
            X_train_balanced, y_train_balanced, X_test, y_test
        )

        comparison_table = trainer.get_comparison_table()
        self.results["model_comparison"] = comparison_table

        # ============================================================
        # STAGE 7: Hyperparameter Tuning (Best Model)
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 7: Hyperparameter Tuning")
        logger.info("=" * 60)

        best_model, best_model_name = trainer.get_best_model()

        tuning_config = self.config.get("tuning", {})
        tuner = HyperparameterTuner(
            method=tuning_config.get("method", "random"),
            n_iter=tuning_config.get("n_iter", 50),
            cv_folds=tuning_config.get("cv_folds", 5),
            scoring=tuning_config.get("scoring", "f1"),
        )

        tuned_model, best_params, best_score = tuner.tune(
            best_model, best_model_name,
            X_train_balanced, y_train_balanced,
        )

        self.results["hyperparameter_tuning"] = {
            "model": best_model_name,
            "best_params": best_params,
            "best_cv_score": float(best_score),
        }

        # ============================================================
        # STAGE 8: Ensemble Building
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 8: Ensemble Building")
        logger.info("=" * 60)

        ensemble_result = {}
        if model_config.get("build_ensemble", True):
            # Get top 3 models
            valid_models = {
                name: result["model"]
                for name, result in training_results.items()
                if "error" not in result
            }
            top_models = dict(
                sorted(
                    valid_models.items(),
                    key=lambda x: training_results[x[0]].get("f1", 0),
                    reverse=True,
                )[:3]
            )

            if len(top_models) >= 2:
                builder = EnsembleBuilder()
                ensemble_result = builder.build_best_ensemble(
                    top_models,
                    X_train_balanced, y_train_balanced,
                    X_test, y_test,
                )

        # ============================================================
        # STAGE 9: Final Model Selection
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 9: Final Model Selection")
        logger.info("=" * 60)

        # Compare tuned model vs ensemble
        from sklearn.metrics import f1_score as f1_metric, roc_auc_score as roc_metric

        tuned_pred = tuned_model.predict(X_test)
        tuned_prob = (
            tuned_model.predict_proba(X_test)[:, 1]
            if hasattr(tuned_model, "predict_proba")
            else tuned_pred.astype(float)
        )
        tuned_f1 = f1_metric(y_test, tuned_pred)

        if ensemble_result and ensemble_result.get("f1", 0) > tuned_f1:
            self.best_model = ensemble_result["model"]
            self.best_model_name = f"ensemble_{ensemble_result['type']}"
            final_pred = ensemble_result["y_pred"]
            final_prob = ensemble_result["y_prob"]
            logger.info(f"Selected: Ensemble ({ensemble_result['type']})")
        else:
            self.best_model = tuned_model
            self.best_model_name = f"tuned_{best_model_name}"
            final_pred = tuned_pred
            final_prob = tuned_prob
            logger.info(f"Selected: Tuned {best_model_name}")

        logger.info(f"Selected: {self.best_model_name} (base: {best_model_name}, CV={best_score:.4f})")

        # ============================================================
        # STAGE 10: Evaluation
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 10: Final Evaluation")
        logger.info("=" * 60)

        evaluator = ModelEvaluator(
            output_dir=self.config["output"]["report_dir"]
        )

        # Evaluate final best model
        final_eval = evaluator.evaluate(
            y_test, final_pred, final_prob, self.best_model_name
        )
        self.results["final_evaluation"] = final_eval

        logger.info(
            f"Accuracy={final_eval.get('accuracy', 0):.4f} | "
            f"Precision={final_eval.get('precision', 0):.4f} | "
            f"Recall={final_eval.get('recall', 0):.4f} | "
            f"F1={final_eval.get('f1_score', 0):.4f} | "
            f"AUC={final_eval.get('roc_auc', 0):.4f}"
        )

        # Evaluate all individual models for comparison
        all_evaluations = {}
        for name, result in training_results.items():
            if "error" not in result:
                eval_result = evaluator.evaluate(
                    y_test, result["y_pred"], result["y_prob"], name
                )
                all_evaluations[name] = eval_result

        self.results["all_evaluations"] = all_evaluations

        # Generate report
        report = evaluator.generate_report(self.best_model_name)
        logger.info(report)

        # Save results
        evaluator.save_results()

        # ============================================================
        # STAGE 11: Explainability (SHAP)
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 11: Model Explainability")
        logger.info("=" * 60)

        eval_config = self.config.get("evaluation", {})
        explainer = ModelExplainer(
            output_dir=self.config["output"]["plot_dir"]
        )

        shap_results = explainer.explain_model(
            self.best_model,
            X_test,
            self.feature_names,
            sample_size=eval_config.get("shap_sample_size", 1000),
            model_name=self.best_model_name,
        )
        self.results["explainability"] = shap_results

        # ============================================================
        # STAGE 12: Visualization
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 12: Generating Visualizations")
        logger.info("=" * 60)

        visualizer = FraudVisualizer(
            output_dir=self.config["output"]["plot_dir"]
        )

        feature_importance = (
            shap_results.get("feature_importance")
            or self.preprocessor.get_feature_importance()
        )

        plots = visualizer.generate_all_plots(
            y_true=y_test,
            evaluation_results=all_evaluations,
            feature_importance=feature_importance,
            comparison_data=comparison_table,
            df=df_for_viz,
        )
        self.results["plots"] = plots

        # ============================================================
        # STAGE 13: Save Model
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 13: Saving Best Model")
        logger.info("=" * 60)

        model_path = self._save_model()
        self.results["model_path"] = model_path

        # Save complete results
        self._save_results()

        # ============================================================
        # SUMMARY
        # ============================================================
        pipeline_time = time.time() - pipeline_start
        self.results["pipeline_time"] = pipeline_time

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {pipeline_time:.1f}s ({pipeline_time/60:.1f}min)")
        logger.info(f"Best model: {self.best_model_name}")
        logger.info(f"Final metrics:")
        logger.info(f"  Accuracy:  {final_eval['accuracy']:.4f}")
        logger.info(f"  Precision: {final_eval['precision']:.4f}")
        logger.info(f"  Recall:    {final_eval['recall']:.4f}")
        logger.info(f"  F1-Score:  {final_eval['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {final_eval['roc_auc']:.4f}")

        return self.results

    def _handle_imbalance(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> tuple:
        """Handle class imbalance using the best technique."""
        imbalance_config = self.config.get("imbalance", {})
        method = imbalance_config.get("method", "auto")
        sampling_strategy = imbalance_config.get("sampling_strategy", 0.3)

        logger.info(f"Handling class imbalance (method: {method})...")
        logger.info(
            f"Before: {len(y_train):,} samples, "
            f"fraud ratio: {y_train.mean():.4%}"
        )

        if method == "auto":
            return self._auto_select_imbalance(X_train, y_train, sampling_strategy)

        return self._apply_imbalance_method(
            method, X_train, y_train, sampling_strategy
        )

    def _auto_select_imbalance(
        self, X: np.ndarray, y: np.ndarray, sampling_strategy: float
    ) -> tuple:
        """Automatically select the best imbalance handling technique."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score as f1_fn
        from sklearn.model_selection import train_test_split as tts

        methods = ["smote", "smoteenn", "adasyn", "class_weight"]
        best_score = 0
        best_method = "smote"
        best_X = X
        best_y = y

        # Quick evaluation split
        X_tr, X_ev, y_tr, y_ev = tts(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        for method in methods:
            try:
                X_balanced, y_balanced, _ = self._apply_imbalance_method(
                    method, X_tr, y_tr, sampling_strategy
                )

                # Quick LR evaluation
                lr = LogisticRegression(
                    max_iter=500, solver="saga", random_state=42, n_jobs=-1
                )
                lr.fit(X_balanced, y_balanced)
                y_pred = lr.predict(X_ev)
                score = f1_fn(y_ev, y_pred)

                logger.info(f"  {method}: F1={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_method = method

            except Exception as e:
                logger.warning(f"  {method}: Failed - {e}")

        logger.info(f"Selected: {best_method} (F1={best_score:.4f})")

        # Apply best method to full training data
        return self._apply_imbalance_method(
            best_method, X, y, sampling_strategy
        )

    def _apply_imbalance_method(
        self, method: str, X: np.ndarray, y: np.ndarray, sampling_strategy: float
    ) -> tuple:
        """Apply a specific imbalance handling technique."""
        if method == "class_weight":
            logger.info("Using class weighting (no resampling)")
            return X, y, "class_weight"

        if method == "smote":
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=42,
            )
        elif method == "smoteenn":
            from imblearn.combine import SMOTEENN
            sampler = SMOTEENN(
                sampling_strategy=sampling_strategy,
                random_state=42,
            )
        elif method == "adasyn":
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(
                sampling_strategy=sampling_strategy,
                random_state=42,
            )
        else:
            logger.warning(f"Unknown method: {method}. Defaulting to SMOTE.")
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=42,
            )
            method = "smote"

        X_resampled, y_resampled = sampler.fit_resample(X, y)
        logger.info(
            f"After {method}: {len(y_resampled):,} samples, "
            f"fraud ratio: {y_resampled.mean():.4%}"
        )

        return X_resampled, y_resampled, method

    def _save_model(self) -> str:
        """Save the best model to disk."""
        model_dir = self.config["output"]["model_dir"]
        model_path = os.path.join(model_dir, "best_model.pkl")

        model_data = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "feature_names": self.feature_names,
            "preprocessor": self.preprocessor,
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {model_path}")
        return model_path

    def _save_results(self) -> str:
        """Save pipeline results to JSON."""
        results_path = os.path.join(
            self.config["output"]["report_dir"], "pipeline_results.json"
        )

        # Make results serializable
        serializable = {}
        for key, value in self.results.items():
            if key in ["plots", "model_path"]:
                serializable[key] = value
            elif isinstance(value, dict):
                serializable[key] = self._make_serializable(value)
            elif isinstance(value, list):
                serializable[key] = [self._make_serializable(v) for v in value]
            else:
                serializable[key] = value

        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        logger.info(f"Results saved to {results_path}")
        return results_path

    def _make_serializable(self, obj):
        """Convert to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    def predict(self, transactions: list) -> list:
        """
        Make predictions on new transactions.

        Args:
            transactions: List of transaction dicts

        Returns:
            List of prediction dicts with fraud probability
        """
        if self.best_model is None:
            raise RuntimeError("No model loaded. Run pipeline first or load a model.")

        import pandas as pd

        df = pd.DataFrame(transactions)
        X, _ = self.preprocessor.transform(df)

        predictions = self.best_model.predict(X)
        probabilities = (
            self.best_model.predict_proba(X)[:, 1]
            if hasattr(self.best_model, "predict_proba")
            else predictions.astype(float)
        )

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "transaction_index": i,
                "is_fraud": bool(pred),
                "fraud_probability": float(prob),
                "risk_level": (
                    "HIGH" if prob > 0.7
                    else "MEDIUM" if prob > 0.3
                    else "LOW"
                ),
            })

        return results
