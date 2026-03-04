"""
Fraud Transaction Detection System - Main Entry Point

A production-grade ML system that automatically selects the best
model and configuration for detecting fraudulent transactions.

Usage:
    python run.py                           # Run with default config
    python run.py --data path/to/data.csv   # Specify dataset path
    python run.py --config path/to/config   # Specify config file
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import FraudDetectionPipeline


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("outputs/pipeline.log", mode="w"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fraud Transaction Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py
  python run.py --data data/transactions.csv
  python run.py --config config/config.yaml --log-level DEBUG
        """,
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to the transaction dataset (CSV)",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Create output directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("FRAUD TRANSACTION DETECTION SYSTEM v1.0.0")
    logger.info("=" * 60)

    try:
        # Initialize and run pipeline
        pipeline = FraudDetectionPipeline(config_path=args.config)
        results = pipeline.run(data_path=args.data)

        # Print summary
        final = results.get("final_evaluation", {})
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)
        print(f"Best Model:  {results.get('best_model_name', pipeline.best_model_name)}")
        print(f"Accuracy:    {final.get('accuracy', 0):.4f} ({final.get('accuracy', 0)*100:.2f}%)")
        print(f"Precision:   {final.get('precision', 0):.4f}")
        print(f"Recall:      {final.get('recall', 0):.4f}")
        print(f"F1-Score:    {final.get('f1_score', 0):.4f}")
        print(f"ROC-AUC:     {final.get('roc_auc', 0):.4f}")
        print(f"Pipeline Time: {results.get('pipeline_time', 0):.1f}s")
        print("=" * 60)
        print(f"\nResults saved to: outputs/reports/")
        print(f"Plots saved to:   outputs/plots/")
        print(f"Model saved to:   outputs/models/best_model.pkl")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        print(f"\nERROR: {e}")
        print("\nPlease download the PaySim dataset:")
        print("  https://www.kaggle.com/datasets/ealaxi/paysim1")
        print("  Place the CSV file in the data/ directory")
        return 1

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\nERROR: Pipeline failed - {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
