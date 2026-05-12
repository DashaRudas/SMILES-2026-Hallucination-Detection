"""
splitting.py — Stratified split by (label, response_length_bin).

Controls for the confounder that hallucinated responses tend to be longer
without re-loading the base language model during splitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


def _approx_response_lengths(df: pd.DataFrame) -> pd.Series:
    """Approximate response length with a light-weight word-count proxy."""
    return df["response"].fillna("").astype(str).str.split().str.len()


def _create_length_bins(response_lengths: pd.Series, n_bins: int = 4) -> pd.Series:
    ranks = response_lengths.rank(method="first")
    return pd.qcut(ranks, q=n_bins, labels=False)


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:

    idx = np.arange(len(y))

    if df is not None:
        response_lengths = _approx_response_lengths(df)
        length_bins = _create_length_bins(response_lengths, n_bins=4)
        stratify_key = df["label"].astype(str) + "_" + length_bins.astype(str)
        # If a combined label/length bucket is too small for k-fold, fall back
        # to label-only stratification instead of failing the split.
        if stratify_key.value_counts().min() < n_splits:
            stratify_key = df["label"].astype(str)
    else:
        stratify_key = pd.Series(y).astype(str)

    # Фиксированный тест-сет со стратификацией по (label, length)
    idx_train_val, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_key,
    )

    # Кросс-валидация по оставшимся 85%
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []

    for tr, va in skf.split(idx_train_val, stratify_key.iloc[idx_train_val]):
        splits.append((
            idx_train_val[tr],   # train indices
            idx_train_val[va],   # val indices (for threshold tuning)
            idx_test             # fixed test set (same for all folds)
        ))

    return splits
