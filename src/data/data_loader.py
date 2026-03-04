"""
Memory-efficient data loading module for large-scale fraud detection datasets.

Supports chunked loading to handle datasets larger than available RAM.
Designed for the PaySim synthetic financial dataset (6.3M+ transactions).
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Memory-efficient data loader with chunked processing."""

    def __init__(self, file_path: str, chunk_size: int = 100000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self._validate_file()

    def _validate_file(self) -> None:
        """Validate that the data file exists and is readable."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Dataset not found at: {self.file_path}\n"
                "Please download the PaySim dataset and place it in the data/ directory.\n"
                "Download from: https://www.kaggle.com/datasets/ealaxi/paysim1"
            )
        file_size_mb = os.path.getsize(self.file_path) / (1024 * 1024)
        logger.info(f"Dataset found: {self.file_path} ({file_size_mb:.1f} MB)")

    def get_dataset_info(self) -> dict:
        """Get basic dataset information without loading full data."""
        logger.info("Scanning dataset for basic statistics...")

        total_rows = 0
        fraud_count = 0
        columns = None
        dtypes = None

        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            if columns is None:
                columns = list(chunk.columns)
                dtypes = chunk.dtypes.to_dict()
            total_rows += len(chunk)
            if "isFraud" in chunk.columns:
                fraud_count += chunk["isFraud"].sum()

        info = {
            "total_rows": total_rows,
            "columns": columns,
            "dtypes": {k: str(v) for k, v in dtypes.items()},
            "fraud_count": int(fraud_count),
            "legitimate_count": total_rows - int(fraud_count),
            "fraud_ratio": fraud_count / total_rows if total_rows > 0 else 0,
        }

        logger.info(
            f"Dataset: {total_rows:,} rows, {len(columns)} columns, "
            f"{fraud_count:,} fraud ({info['fraud_ratio']:.4%})"
        )
        return info

    def load_chunked(self, columns: Optional[list] = None) -> pd.DataFrame:
        """Load the full dataset in chunks, optionally selecting specific columns."""
        logger.info(f"Loading dataset in chunks of {self.chunk_size:,} rows...")

        chunks = []
        for i, chunk in enumerate(
            pd.read_csv(self.file_path, chunksize=self.chunk_size, usecols=columns)
        ):
            chunks.append(chunk)
            if (i + 1) % 10 == 0:
                rows_loaded = (i + 1) * self.chunk_size
                logger.info(f"  Loaded {rows_loaded:,} rows...")

        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
        self._optimize_memory(df)
        return df

    def load_fraud_and_sample(
        self, max_rows: int = 1000000, fraud_multiplier: float = 3.0
    ) -> pd.DataFrame:
        """
        Load ALL fraud transactions first, then sample legitimate ones.
        This ensures no fraud cases are lost during sampling.
        """
        logger.info("Loading with fraud-priority strategy...")

        fraud_chunks = []
        legit_chunks = []
        total_fraud = 0
        total_legit = 0

        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            fraud_mask = chunk["isFraud"] == 1
            fraud_chunks.append(chunk[fraud_mask])
            legit_chunks.append(chunk[~fraud_mask])
            total_fraud += fraud_mask.sum()
            total_legit += (~fraud_mask).sum()

        all_fraud = pd.concat(fraud_chunks, ignore_index=True)
        all_legit = pd.concat(legit_chunks, ignore_index=True)

        logger.info(
            f"Found {len(all_fraud):,} fraud and {len(all_legit):,} legitimate transactions"
        )

        # Calculate how many legitimate samples to keep
        legit_budget = max_rows - len(all_fraud)
        if legit_budget < len(all_legit):
            all_legit = all_legit.sample(n=legit_budget, random_state=42)
            logger.info(f"Sampled {legit_budget:,} legitimate transactions")

        df = pd.concat([all_fraud, all_legit], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(
            f"Final dataset: {len(df):,} rows "
            f"(fraud ratio: {len(all_fraud)/len(df):.4%})"
        )
        self._optimize_memory(df)
        return df

    def _optimize_memory(self, df: pd.DataFrame) -> None:
        """Optimize DataFrame memory usage by downcasting numeric types."""
        initial_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)

        for col in df.select_dtypes(include=["int64"]).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= 0:
                if col_max <= np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif col_max <= np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif col_max <= np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
            else:
                if (
                    col_min >= np.iinfo(np.int8).min
                    and col_max <= np.iinfo(np.int8).max
                ):
                    df[col] = df[col].astype(np.int8)
                elif (
                    col_min >= np.iinfo(np.int16).min
                    and col_max <= np.iinfo(np.int16).max
                ):
                    df[col] = df[col].astype(np.int16)
                elif (
                    col_min >= np.iinfo(np.int32).min
                    and col_max <= np.iinfo(np.int32).max
                ):
                    df[col] = df[col].astype(np.int32)

        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype(np.float32)

        final_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        reduction = (1 - final_memory / initial_memory) * 100
        logger.info(
            f"Memory optimized: {initial_memory:.1f}MB -> {final_memory:.1f}MB "
            f"({reduction:.1f}% reduction)"
        )
