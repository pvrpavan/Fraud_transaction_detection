"""
Memory-efficient data loading module for large-scale fraud detection datasets.

Supports chunked loading to handle datasets larger than available RAM.
Designed for the PaySim synthetic financial dataset (6.3M+ transactions).

Features:
    - Fraud-preserving sampling: ALL fraud transactions are retained
    - Flexible row selection via start_row / max_rows
    - Memory-efficient chunk-based reading with pd.read_csv(chunksize=...)
"""

import logging
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Memory-efficient data loader with chunked processing and fraud-preserving sampling."""

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 100_000,
        max_rows: int = 1_000_000,
        start_row: int = 0,
        target_fraud_ratio: float = 0.0,
    ):
        """
        Initialize DataLoader.

        Args:
            file_path: Path to the CSV dataset.
            chunk_size: Number of rows per chunk when reading.
            max_rows: Maximum total rows in the final dataset.
            start_row: Row offset to start reading from (for flexible windowing).
            target_fraud_ratio: Target fraud ratio in the final dataset.
                0.0  = keep all legit up to max_rows (default, fraud-preserving).
                >0   = dynamically reduce legit rows so that
                       fraud / total ~= target_fraud_ratio.
                       E.g. 0.5 gives roughly 50/50 fraud vs legit.
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.max_rows = max_rows
        self.start_row = start_row
        self.target_fraud_ratio = target_fraud_ratio
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

        for chunk in pd.read_csv(
            self.file_path, chunksize=self.chunk_size, on_bad_lines="skip"
        ):
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

    def load_fraud_and_sample(
        self,
        max_rows: Optional[int] = None,
        start_row: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fraud-preserving sampling: load ALL fraud transactions first,
        then fill the remaining budget with legitimate transactions.

        Two-pass approach (memory-efficient):
            Pass 1 - Collect every fraud row from the entire file.
            Pass 2 - Stream legitimate rows (respecting start_row offset)
                     until the budget (max_rows - fraud_count) is filled.

        Args:
            max_rows: Override for self.max_rows.
            start_row: Override for self.start_row.

        Returns:
            DataFrame with all fraud rows + sampled legitimate rows,
            shuffled, with total size <= max_rows.
        """
        max_rows = max_rows or self.max_rows
        start_row = start_row if start_row is not None else self.start_row

        logger.info(
            f"Loading with fraud-preserving strategy "
            f"(max_rows={max_rows:,}, start_row={start_row:,}) ..."
        )

        # ---- Pass 1: collect ALL fraud transactions (full file scan) ----
        fraud_chunks = []

        for chunk in pd.read_csv(
            self.file_path, chunksize=self.chunk_size, on_bad_lines="skip"
        ):
            fraud_in_chunk = chunk[chunk["isFraud"] == 1]
            if len(fraud_in_chunk) > 0:
                fraud_chunks.append(fraud_in_chunk)

        all_fraud = (
            pd.concat(fraud_chunks, ignore_index=True)
            if fraud_chunks
            else pd.DataFrame()
        )
        total_fraud = len(all_fraud)
        logger.info(f"Pass 1 complete: collected {total_fraud:,} fraud transactions")

        # ---- Fraud safety check ----
        if total_fraud < 2000:
            warnings.warn(
                f"Only {total_fraud} fraud samples found (< 2000). "
                "Model performance may be degraded. Consider using a larger "
                "dataset or adjusting the fraud detection threshold.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                f"LOW FRAUD WARNING: only {total_fraud} fraud samples found "
                "(threshold: 2000). Proceeding with all available fraud rows."
            )

        # ---- Calculate legitimate budget ----
        target_ratio = self.target_fraud_ratio
        if target_ratio > 0 and total_fraud > 0:
            # Dynamic balancing: adjust legit count to achieve target fraud ratio
            # fraud / (fraud + legit) = target_ratio
            # => legit = fraud * (1 - target_ratio) / target_ratio
            dynamic_legit = int(total_fraud * (1.0 - target_ratio) / target_ratio)
            legit_budget = min(dynamic_legit, max(0, max_rows - total_fraud))
            logger.info(
                f"Dynamic balancing: target_fraud_ratio={target_ratio:.2%} -> "
                f"legit_budget={legit_budget:,} (fraud={total_fraud:,})"
            )
        else:
            legit_budget = max(0, max_rows - total_fraud)
            logger.info(
                f"Legitimate budget: {legit_budget:,} "
                f"(max_rows={max_rows:,} - fraud={total_fraud:,})"
            )

        # ---- Pass 2: stream legitimate transactions ----
        legit_chunks = []
        legit_collected = 0
        rows_seen = 0

        reader = pd.read_csv(
            self.file_path, chunksize=self.chunk_size, on_bad_lines="skip"
        )
        with reader:
            for chunk in reader:
                chunk_end = rows_seen + len(chunk)

                # Skip chunks entirely before start_row
                if chunk_end <= start_row:
                    rows_seen = chunk_end
                    continue

                # Slice the chunk if it partially overlaps with start_row
                if rows_seen < start_row:
                    offset_in_chunk = start_row - rows_seen
                    chunk = chunk.iloc[offset_in_chunk:]

                rows_seen = chunk_end

                legit_chunk = chunk[chunk["isFraud"] == 0]
                if len(legit_chunk) == 0:
                    continue

                needed = legit_budget - legit_collected
                if needed <= 0:
                    break

                if len(legit_chunk) <= needed:
                    legit_chunks.append(legit_chunk)
                    legit_collected += len(legit_chunk)
                else:
                    legit_chunks.append(
                        legit_chunk.sample(n=needed, random_state=42)
                    )
                    legit_collected += needed
                    break

        all_legit = (
            pd.concat(legit_chunks, ignore_index=True)
            if legit_chunks
            else pd.DataFrame()
        )
        logger.info(
            f"Pass 2 complete: collected {len(all_legit):,} legitimate transactions"
        )

        # ---- Combine and shuffle ----
        df = pd.concat([all_fraud, all_legit], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Final trim - keep all fraud, trim only legit if needed
        if len(df) > max_rows:
            fraud_part = df[df["isFraud"] == 1]
            legit_part = df[df["isFraud"] == 0].sample(
                n=max_rows - len(fraud_part), random_state=42
            )
            df = pd.concat([fraud_part, legit_part], ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        fraud_in_final = int(df["isFraud"].sum())
        legit_in_final = len(df) - fraud_in_final
        logger.info(
            f"Final dataset: {len(df):,} rows "
            f"(fraud={fraud_in_final:,}, legit={legit_in_final:,}, "
            f"fraud_ratio={fraud_in_final / len(df):.4%})"
        )

        self._optimize_memory(df)
        return df

    def load_chunked(self, columns: Optional[list] = None) -> pd.DataFrame:
        """Load the full dataset in chunks, optionally selecting specific columns."""
        logger.info(f"Loading dataset in chunks of {self.chunk_size:,} rows...")

        chunks = []
        for i, chunk in enumerate(
            pd.read_csv(
                self.file_path, chunksize=self.chunk_size, usecols=columns
            )
        ):
            chunks.append(chunk)
            if (i + 1) % 10 == 0:
                rows_loaded = (i + 1) * self.chunk_size
                logger.info(f"  Loaded {rows_loaded:,} rows...")

        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
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
        reduction = (
            (1 - final_memory / initial_memory) * 100 if initial_memory > 0 else 0
        )
        logger.info(
            f"Memory optimized: {initial_memory:.1f}MB -> {final_memory:.1f}MB "
            f"({reduction:.1f}% reduction)"
        )
