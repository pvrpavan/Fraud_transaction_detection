"""
Data preprocessing and feature engineering module.

Handles feature creation, encoding, scaling, selection, and
dimensionality reduction for the fraud detection pipeline.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Automated feature engineering and preprocessing pipeline."""

    def __init__(
        self,
        scaling_method: str = "robust",
        feature_selection: bool = True,
        variance_threshold: float = 0.01,
    ):
        self.scaling_method = scaling_method
        self.feature_selection = feature_selection
        self.variance_threshold = variance_threshold
        self.scaler = None
        self.label_encoders: dict = {}
        self.selected_features: Optional[list] = None
        self.feature_importance: Optional[dict] = None
        self._is_fitted = False
        self._type_amount_stats: dict = {}

    def fit_transform(
        self, df: pd.DataFrame, target_col: str = "isFraud"
    ) -> tuple:
        """
        Fit the preprocessor and transform the data.

        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Starting preprocessing pipeline...")

        df = df.copy()

        # Step 1: Clean data
        df = self._clean_data(df, target_col)

        # Step 2: Feature engineering
        df = self._engineer_features(df)

        # Step 3: Encode categorical variables
        df = self._encode_categoricals(df)

        # Step 4: Separate features and target
        y = df[target_col].values
        drop_cols = [target_col]
        if "isFlaggedFraud" in df.columns:
            drop_cols.append("isFlaggedFraud")
        X = df.drop(columns=drop_cols, errors="ignore")

        # Step 5: Handle remaining missing values
        X = X.fillna(0)

        # Step 6: Feature scaling
        X = self._scale_features(X, fit=True)

        # Step 7: Feature selection
        if self.feature_selection:
            X = self._select_features(X, y)

        self._is_fitted = True
        feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [
            f"feature_{i}" for i in range(X.shape[1])
        ]

        logger.info(
            f"Preprocessing complete: {X.shape[0]:,} samples, "
            f"{X.shape[1]} features"
        )

        return X.values if isinstance(X, pd.DataFrame) else X, y, feature_names

    def transform(self, df: pd.DataFrame, target_col: str = "isFraud") -> tuple:
        """Transform new data using fitted preprocessor."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        df = df.copy()
        df = self._clean_data(df, target_col)
        df = self._engineer_features(df)
        df = self._encode_categoricals(df, fit=False)

        y = df[target_col].values if target_col in df.columns else None
        drop_cols = [target_col]
        if "isFlaggedFraud" in df.columns:
            drop_cols.append("isFlaggedFraud")
        X = df.drop(columns=drop_cols, errors="ignore")
        X = X.fillna(0)
        X = self._scale_features(X, fit=False)

        if self.selected_features is not None:
            available = [f for f in self.selected_features if f in X.columns]
            X = X[available]

        return X.values if isinstance(X, pd.DataFrame) else X, y

    def _clean_data(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Clean and validate the dataset."""
        initial_rows = len(df)

        # Drop duplicates
        df = df.drop_duplicates()

        # Drop columns with too many missing values (>50%)
        missing_ratio = df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
        if cols_to_drop:
            logger.info(f"Dropping high-missing columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        # Drop identifier columns that don't help prediction
        id_cols = ["nameOrig", "nameDest"]
        existing_id_cols = [c for c in id_cols if c in df.columns]
        if existing_id_cols:
            df = df.drop(columns=existing_id_cols)

        # Drop post-transaction balance columns to prevent data leakage.
        # In real-world fraud detection, predictions must be made BEFORE
        # the transaction is processed, so post-transaction balances
        # would not be available at prediction time.
        post_txn_cols = ["newbalanceOrig", "newbalanceDest"]
        existing_post_cols = [c for c in post_txn_cols if c in df.columns]
        if existing_post_cols:
            logger.info(f"Dropping post-transaction columns: {existing_post_cols}")
            df = df.drop(columns=existing_post_cols)

        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            logger.info(f"Cleaned {rows_removed:,} duplicate rows")

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing columns."""
        logger.info("Engineering features...")

        # Amount relative features
        if "amount" in df.columns:
            # Log transform of amount (handles skewness)
            df["amount_log"] = np.log1p(df["amount"])

            # Sqrt transform (moderate skew reduction)
            df["amount_sqrt"] = np.sqrt(df["amount"])

        # NOTE: Balance error features (orig_balance_error, dest_balance_error,
        # orig_error_flag, dest_error_flag) were removed because they cause
        # data leakage in the PaySim synthetic dataset. The simulator generates
        # balance discrepancies only for fraudulent transactions, making these
        # features near-perfect proxies for the fraud label.

        # Transaction type interaction features
        if "type" in df.columns and "amount" in df.columns:
            if not self._type_amount_stats:
                # During training: compute and store per-type stats
                type_stats = df.groupby("type")["amount"].agg(["mean", "std"])
                self._type_amount_stats = {
                    t: {"mean": row["mean"], "std": row["std"]}
                    for t, row in type_stats.iterrows()
                }

            # Use stored stats for both training and prediction
            type_means = df["type"].map(
                lambda t: self._type_amount_stats.get(t, {}).get("mean", df["amount"].mean())
            )
            type_stds = df["type"].map(
                lambda t: self._type_amount_stats.get(t, {}).get("std", 1.0)
            )
            df["amount_vs_type_mean"] = df["amount"] / (type_means + 1)
            df["amount_zscore_by_type"] = np.where(
                type_stds > 0,
                (df["amount"] - type_means) / type_stds,
                0,
            )

        # Time-based patterns (potential for fraud spikes)
        if "step" in df.columns:
            df["hour_of_day"] = df["step"] % 24
            df["day_of_sim"] = df["step"] // 24
            # Fraud often happens at odd hours
            df["unusual_hour"] = (
                (df["hour_of_day"] < 6) | (df["hour_of_day"] > 22)
            ).astype(int)
            # Cyclical encoding for hour (captures periodicity)
            df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

        # Round amount detection (fraudsters often use round numbers)
        if "amount" in df.columns:
            df["round_amount"] = (df["amount"] == df["amount"].round()).astype(int)

        # Replace infinities
        df = df.replace([np.inf, -np.inf], 0)

        n_new_features = len(df.columns) - 11  # Approximate original columns
        logger.info(f"Created {max(0, n_new_features)} new features")

        return df

    def _encode_categoricals(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Encode categorical variables using label encoding."""
        categorical_cols = df.select_dtypes(include=["object", "category", "str"]).columns

        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    known_classes = set(le.classes_)
                    df[col] = df[col].astype(str).map(
                        lambda x, kc=known_classes, encoder=le: (
                            encoder.transform([x])[0] if x in kc else -1
                        )
                    )
                else:
                    df[col] = 0

        return df

    def _scale_features(
        self, X: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Scale numeric features."""
        if fit:
            if self.scaling_method == "standard":
                self.scaler = StandardScaler()
            elif self.scaling_method == "minmax":
                self.scaler = MinMaxScaler()
            elif self.scaling_method == "robust":
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling_method}")

            scaled_values = self.scaler.fit_transform(X)
        else:
            scaled_values = self.scaler.transform(X)

        return pd.DataFrame(scaled_values, columns=X.columns, index=X.index)

    def _select_features(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> pd.DataFrame:
        """Select most important features using variance threshold and mutual information."""
        logger.info("Performing feature selection...")
        initial_features = X.shape[1]

        # Step 1: Remove low-variance features
        selector = VarianceThreshold(threshold=self.variance_threshold)
        mask = selector.fit(X).get_support()
        selected_cols = X.columns[mask].tolist()
        X = X[selected_cols]

        # Step 2: Mutual information-based selection
        if X.shape[1] > 5:
            mi_scores = mutual_info_classif(
                X.values, y, random_state=42, n_neighbors=5
            )
            mi_df = pd.DataFrame(
                {"feature": X.columns, "mi_score": mi_scores}
            ).sort_values("mi_score", ascending=False)

            self.feature_importance = dict(
                zip(mi_df["feature"], mi_df["mi_score"])
            )

            # Keep features with MI score > 0
            important_features = mi_df[mi_df["mi_score"] > 0]["feature"].tolist()
            if len(important_features) >= 5:
                X = X[important_features]

        self.selected_features = list(X.columns)

        logger.info(
            f"Feature selection: {initial_features} -> {X.shape[1]} features"
        )
        return X

    def get_feature_importance(self) -> Optional[dict]:
        """Return feature importance scores from mutual information analysis."""
        return self.feature_importance
