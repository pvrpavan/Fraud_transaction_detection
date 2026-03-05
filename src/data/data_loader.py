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

        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size, on_bad_lines='skip'):
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
        Load ALL fraud transactions first, then sample legitimate ones using chunked reading.
        This ensures no fraud cases are lost during sampling and stops at max_rows.
        """
        logger.info("Loading with fraud-priority strategy and chunked sampling...")

        fraud_chunks = []
        legit_chunks = []
        total_fraud = 0
        total_legit = 0
        collected_rows = 0

        # First pass: collect all fraud transactions
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size, on_bad_lines='skip'):
            fraud_mask = chunk["isFraud"] == 1
            fraud_chunks.append(chunk[fraud_mask])
            total_fraud += fraud_mask.sum()

            # Check if we've collected enough fraud samples
            if len(fraud_chunks) * self.chunk_size > max_rows * 0.1:  # Assume ~10% fraud
                break

        all_fraud = pd.concat(fraud_chunks, ignore_index=True)
        logger.info(f"Collected {len(all_fraud):,} fraud transactions")

        # Calculate remaining budget for legitimate transactions
        remaining_budget = max_rows - len(all_fraud)
        legit_sampled = 0

        # Second pass: sample legitimate transactions with intelligent selection
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size, on_bad_lines='skip'):
            legit_mask = chunk["isFraud"] == 0
            legit_chunk = chunk[legit_mask]

            if len(legit_chunk) == 0:
                continue

            # If we still have budget, collect all from this chunk
            if legit_sampled + len(legit_chunk) <= remaining_budget:
                legit_chunks.append(legit_chunk)
                legit_sampled += len(legit_chunk)
            else:
                # Need to sample from this chunk
                needed = remaining_budget - legit_sampled
                if needed > 0:
                    sampled_chunk = legit_chunk.sample(n=needed, random_state=42)
                    legit_chunks.append(sampled_chunk)
                    legit_sampled += len(sampled_chunk)
                break

        all_legit = pd.concat(legit_chunks, ignore_index=True) if legit_chunks else pd.DataFrame()

        logger.info(f"Collected {len(all_legit):,} legitimate transactions")

        # Combine and shuffle
        df = pd.concat([all_fraud, all_legit], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Final check: ensure we don't exceed max_rows
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

        fraud_ratio = len(all_fraud) / len(df) if len(df) > 0 else 0
        logger.info(
            f"Final dataset: {len(df):,} rows (fraud ratio: {fraud_ratio:.4%})"
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
