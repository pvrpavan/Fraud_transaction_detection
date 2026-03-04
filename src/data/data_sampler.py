"""
Intelligent data sampling module for fraud detection.

Implements multiple sampling strategies to select the most informative
subset of transactions while preserving fraud patterns.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class DataSampler:
    """Intelligent data sampler with multiple strategies."""

    def __init__(self, max_rows: int = 1000000, random_state: int = 42):
        self.max_rows = max_rows
        self.random_state = random_state

    def sample(self, df: pd.DataFrame, strategy: str = "hybrid") -> pd.DataFrame:
        """
        Apply the specified sampling strategy.

        Args:
            df: Full dataset
            strategy: One of 'stratified', 'fraud_focused', 'clustering',
                      'anomaly_focused', 'hybrid'

        Returns:
            Sampled DataFrame
        """
        if len(df) <= self.max_rows:
            logger.info(
                f"Dataset ({len(df):,} rows) fits within budget. No sampling needed."
            )
            return df

        logger.info(
            f"Applying '{strategy}' sampling: {len(df):,} -> {self.max_rows:,} rows"
        )

        strategy_map = {
            "stratified": self._stratified_sample,
            "fraud_focused": self._fraud_focused_sample,
            "clustering": self._clustering_sample,
            "anomaly_focused": self._anomaly_focused_sample,
            "hybrid": self._hybrid_sample,
        }

        if strategy not in strategy_map:
            raise ValueError(
                f"Unknown strategy: {strategy}. Choose from {list(strategy_map.keys())}"
            )

        sampled = strategy_map[strategy](df)
        logger.info(
            f"Sampling complete: {len(sampled):,} rows "
            f"(fraud: {sampled['isFraud'].sum():,}, "
            f"ratio: {sampled['isFraud'].mean():.4%})"
        )
        return sampled

    def _stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stratified sampling preserving fraud ratio and transaction type distribution."""
        # Keep ALL fraud cases
        fraud_df = df[df["isFraud"] == 1].copy()
        legit_df = df[df["isFraud"] == 0].copy()

        legit_budget = self.max_rows - len(fraud_df)

        # Stratify legitimate transactions by transaction type
        if "type" in legit_df.columns:
            type_counts = legit_df["type"].value_counts(normalize=True)
            sampled_legit_parts = []
            for txn_type, ratio in type_counts.items():
                type_df = legit_df[legit_df["type"] == txn_type]
                n_samples = min(int(legit_budget * ratio), len(type_df))
                sampled_legit_parts.append(
                    type_df.sample(n=n_samples, random_state=self.random_state)
                )
            sampled_legit = pd.concat(sampled_legit_parts, ignore_index=True)
        else:
            sampled_legit = legit_df.sample(
                n=legit_budget, random_state=self.random_state
            )

        result = pd.concat([fraud_df, sampled_legit], ignore_index=True)
        return result.sample(frac=1, random_state=self.random_state).reset_index(
            drop=True
        )

    def _fraud_focused_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Focus sampling on transactions similar to known fraud patterns."""
        fraud_df = df[df["isFraud"] == 1].copy()
        legit_df = df[df["isFraud"] == 0].copy()

        legit_budget = self.max_rows - len(fraud_df)

        # Compute numeric features for similarity
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["isFraud", "isFlaggedFraud"]]

        if len(numeric_cols) > 0:
            # Score legitimate transactions by similarity to fraud patterns
            fraud_means = fraud_df[numeric_cols].mean()
            fraud_stds = fraud_df[numeric_cols].std().replace(0, 1)

            # Compute normalized distance to fraud centroid
            legit_features = legit_df[numeric_cols].fillna(0)
            distances = ((legit_features - fraud_means) / fraud_stds).pow(2).sum(axis=1)

            # Keep transactions closest to fraud patterns (most informative)
            # Plus some random ones for diversity
            n_close = int(legit_budget * 0.7)
            n_random = legit_budget - n_close

            closest_indices = distances.nsmallest(n_close).index
            remaining = legit_df.drop(closest_indices)
            random_indices = remaining.sample(
                n=min(n_random, len(remaining)), random_state=self.random_state
            ).index

            sampled_legit = legit_df.loc[
                closest_indices.tolist() + random_indices.tolist()
            ]
        else:
            sampled_legit = legit_df.sample(
                n=legit_budget, random_state=self.random_state
            )

        result = pd.concat([fraud_df, sampled_legit], ignore_index=True)
        return result.sample(frac=1, random_state=self.random_state).reset_index(
            drop=True
        )

    def _clustering_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clustering-based sampling to preserve data distribution diversity."""
        fraud_df = df[df["isFraud"] == 1].copy()
        legit_df = df[df["isFraud"] == 0].copy()

        legit_budget = self.max_rows - len(fraud_df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["isFraud", "isFlaggedFraud"]]

        if len(numeric_cols) > 0:
            # Use MiniBatchKMeans for memory efficiency
            n_clusters = min(100, legit_budget // 100)
            scaler = StandardScaler()

            legit_features = scaler.fit_transform(
                legit_df[numeric_cols].fillna(0).values
            )

            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                batch_size=10000,
                n_init=3,
            )
            clusters = kmeans.fit_predict(legit_features)
            legit_df = legit_df.copy()
            legit_df["_cluster"] = clusters

            # Sample proportionally from each cluster
            sampled_parts = []
            cluster_counts = pd.Series(clusters).value_counts()
            for cluster_id, count in cluster_counts.items():
                cluster_df = legit_df[legit_df["_cluster"] == cluster_id]
                n_samples = max(1, int(legit_budget * count / len(legit_df)))
                n_samples = min(n_samples, len(cluster_df))
                sampled_parts.append(
                    cluster_df.sample(n=n_samples, random_state=self.random_state)
                )

            sampled_legit = pd.concat(sampled_parts, ignore_index=True)
            if "_cluster" in sampled_legit.columns:
                sampled_legit = sampled_legit.drop(columns=["_cluster"])
        else:
            sampled_legit = legit_df.sample(
                n=legit_budget, random_state=self.random_state
            )

        # Trim to exact budget
        if len(sampled_legit) > legit_budget:
            sampled_legit = sampled_legit.sample(
                n=legit_budget, random_state=self.random_state
            )

        result = pd.concat([fraud_df, sampled_legit], ignore_index=True)
        return result.sample(frac=1, random_state=self.random_state).reset_index(
            drop=True
        )

    def _anomaly_focused_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use Isolation Forest to keep anomalous (potentially fraudulent) transactions."""
        fraud_df = df[df["isFraud"] == 1].copy()
        legit_df = df[df["isFraud"] == 0].copy()

        legit_budget = self.max_rows - len(fraud_df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["isFraud", "isFlaggedFraud"]]

        if len(numeric_cols) > 0:
            # Use a subsample for fitting Isolation Forest (memory efficient)
            fit_sample = legit_df.sample(
                n=min(50000, len(legit_df)), random_state=self.random_state
            )

            iso_forest = IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=self.random_state,
                n_jobs=-1,
            )
            iso_forest.fit(fit_sample[numeric_cols].fillna(0))

            # Score all legitimate transactions
            scores = iso_forest.decision_function(legit_df[numeric_cols].fillna(0))
            legit_df = legit_df.copy()
            legit_df["_anomaly_score"] = scores

            # Keep most anomalous (lowest scores) + some normal ones
            n_anomalous = int(legit_budget * 0.6)
            n_normal = legit_budget - n_anomalous

            anomalous = legit_df.nsmallest(n_anomalous, "_anomaly_score")
            normal_pool = legit_df.drop(anomalous.index)
            normal = normal_pool.sample(
                n=min(n_normal, len(normal_pool)), random_state=self.random_state
            )

            sampled_legit = pd.concat([anomalous, normal], ignore_index=True)
            if "_anomaly_score" in sampled_legit.columns:
                sampled_legit = sampled_legit.drop(columns=["_anomaly_score"])
        else:
            sampled_legit = legit_df.sample(
                n=legit_budget, random_state=self.random_state
            )

        result = pd.concat([fraud_df, sampled_legit], ignore_index=True)
        return result.sample(frac=1, random_state=self.random_state).reset_index(
            drop=True
        )

    def _hybrid_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hybrid strategy combining stratified + fraud-focused + anomaly sampling.
        This is the recommended default strategy.
        """
        fraud_df = df[df["isFraud"] == 1].copy()
        legit_df = df[df["isFraud"] == 0].copy()

        legit_budget = self.max_rows - len(fraud_df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["isFraud", "isFlaggedFraud"]]

        # Split budget: 40% fraud-similar, 30% anomalous, 30% stratified random
        n_fraud_similar = int(legit_budget * 0.4)
        n_anomalous = int(legit_budget * 0.3)
        n_stratified = legit_budget - n_fraud_similar - n_anomalous

        selected_indices = set()

        # Part 1: Fraud-similar transactions
        if len(numeric_cols) > 0:
            fraud_means = fraud_df[numeric_cols].mean()
            fraud_stds = fraud_df[numeric_cols].std().replace(0, 1)
            legit_features = legit_df[numeric_cols].fillna(0)
            distances = ((legit_features - fraud_means) / fraud_stds).pow(2).sum(axis=1)
            closest = distances.nsmallest(n_fraud_similar).index
            selected_indices.update(closest.tolist())

        # Part 2: Anomalous transactions
        if len(numeric_cols) > 0:
            remaining = legit_df.drop(index=list(selected_indices), errors="ignore")
            if len(remaining) > 0:
                fit_sample = remaining.sample(
                    n=min(30000, len(remaining)), random_state=self.random_state
                )
                iso = IsolationForest(
                    n_estimators=50,
                    contamination=0.1,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                iso.fit(fit_sample[numeric_cols].fillna(0))
                scores = iso.decision_function(remaining[numeric_cols].fillna(0))
                anomalous_idx = (
                    pd.Series(scores, index=remaining.index)
                    .nsmallest(n_anomalous)
                    .index
                )
                selected_indices.update(anomalous_idx.tolist())

        # Part 3: Stratified random from remaining
        remaining = legit_df.drop(index=list(selected_indices), errors="ignore")
        if "type" in remaining.columns and len(remaining) > 0:
            type_ratios = remaining["type"].value_counts(normalize=True)
            for txn_type, ratio in type_ratios.items():
                type_df = remaining[remaining["type"] == txn_type]
                n = min(int(n_stratified * ratio), len(type_df))
                if n > 0:
                    sampled = type_df.sample(n=n, random_state=self.random_state)
                    selected_indices.update(sampled.index.tolist())
        elif len(remaining) > 0:
            sampled = remaining.sample(
                n=min(n_stratified, len(remaining)), random_state=self.random_state
            )
            selected_indices.update(sampled.index.tolist())

        sampled_legit = legit_df.loc[list(selected_indices)]

        # Ensure we don't exceed budget
        if len(sampled_legit) > legit_budget:
            sampled_legit = sampled_legit.sample(
                n=legit_budget, random_state=self.random_state
            )

        result = pd.concat([fraud_df, sampled_legit], ignore_index=True)
        return result.sample(frac=1, random_state=self.random_state).reset_index(
            drop=True
        )

    def evaluate_sampling_strategies(
        self, df: pd.DataFrame
    ) -> dict:
        """Evaluate all strategies and recommend the best one."""
        logger.info("Evaluating sampling strategies...")

        results = {}
        strategies = [
            "stratified",
            "fraud_focused",
            "clustering",
            "anomaly_focused",
            "hybrid",
        ]

        for strategy in strategies:
            try:
                sampled = self.sample(df, strategy=strategy)
                fraud_ratio = sampled["isFraud"].mean()
                fraud_count = sampled["isFraud"].sum()

                # Compute diversity score based on feature variance
                numeric_cols = sampled.select_dtypes(include=[np.number]).columns
                numeric_cols = [
                    c for c in numeric_cols if c not in ["isFraud", "isFlaggedFraud"]
                ]
                diversity = sampled[numeric_cols].std().mean() if len(numeric_cols) > 0 else 0

                results[strategy] = {
                    "total_rows": len(sampled),
                    "fraud_count": int(fraud_count),
                    "fraud_ratio": float(fraud_ratio),
                    "diversity_score": float(diversity),
                }
                logger.info(
                    f"  {strategy}: {len(sampled):,} rows, "
                    f"fraud={fraud_count:,} ({fraud_ratio:.4%}), "
                    f"diversity={diversity:.4f}"
                )
            except Exception as e:
                logger.warning(f"  {strategy}: Failed - {e}")
                results[strategy] = {"error": str(e)}

        return results
