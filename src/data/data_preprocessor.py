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
            # Extract useful info before dropping
            df = df.drop(columns=existing_id_cols)

        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            logger.info(f"Cleaned {rows_removed:,} duplicate rows")

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing columns."""
        logger.info("Engineering features...")

        # Balance change features
        if "oldbalanceOrg" in df.columns and "newbalanceOrig" in df.columns:
            df["balance_change_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
            df["balance_change_ratio_orig"] = np.where(
                df["oldbalanceOrg"] > 0,
                df["balance_change_orig"] / df["oldbalanceOrg"],
                0,
            )

        if "oldbalanceDest" in df.columns and "newbalanceDest" in df.columns:
            df["balance_change_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
            df["balance_change_ratio_dest"] = np.where(
                df["oldbalanceDest"] > 0,
                df["balance_change_dest"] / df["oldbalanceDest"],
                0,
            )

        # Amount relative features
        if "amount" in df.columns:
            if "oldbalanceOrg" in df.columns:
                df["amount_to_balance_ratio"] = np.where(
                    df["oldbalanceOrg"] > 0,
                    df["amount"] / df["oldbalanceOrg"],
                    0,
                )

            # Log transform of amount (handles skewness)
            df["amount_log"] = np.log1p(df["amount"])

        # Error/discrepancy features (strong fraud indicators)
        if all(
            c in df.columns
            for c in ["oldbalanceOrg", "amount", "newbalanceOrig"]
        ):
            df["orig_balance_error"] = (
                df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
            )
            df["orig_error_flag"] = (df["orig_balance_error"].abs() > 1).astype(int)

        if all(
            c in df.columns
            for c in ["oldbalanceDest", "amount", "newbalanceDest"]
        ):
            df["dest_balance_error"] = (
                df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
            )
            df["dest_error_flag"] = (df["dest_balance_error"].abs() > 1).astype(int)

        # Zero balance flags
        if "oldbalanceOrg" in df.columns:
            df["zero_balance_orig"] = (df["oldbalanceOrg"] == 0).astype(int)
        if "newbalanceOrig" in df.columns:
            df["emptied_account"] = (
                (df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)
            ).astype(int)

        # Transaction type interaction features
        if "type" in df.columns and "amount" in df.columns:
            type_amount_mean = df.groupby("type")["amount"].transform("mean")
            df["amount_vs_type_mean"] = df["amount"] / (type_amount_mean + 1)

        # Step features (time-based)
        if "step" in df.columns:
            df["hour_of_day"] = df["step"] % 24
            df["day_of_sim"] = df["step"] // 24

        # Replace infinities
        df = df.replace([np.inf, -np.inf], 0)

        n_new_features = len(df.columns) - 11  # Approximate original columns
        logger.info(f"Created {max(0, n_new_features)} new features")

        return df

    def _encode_categoricals(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Encode categorical variables using label encoding."""
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

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
