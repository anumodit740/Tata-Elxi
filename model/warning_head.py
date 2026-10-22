"""Frozen-v3 near-failure classifier and train-OOF calibration helpers.

The head predicts a supervised ``true RUL <= N`` target. It does not identify
physical fault onset. This module never opens raw data, test labels or files.
Its caller must supply features from preprocessing fitted on training engines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve

from core import engine_weights, feature_columns


HORIZONS = (20, 30, 50)


def _binary_arrays(y_true, probability) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true)
    scores = np.asarray(probability, dtype=float)
    if truth.ndim != 1 or scores.ndim != 1 or len(truth) != len(scores) or not len(truth):
        raise ValueError("Labels and probabilities must be nonempty, aligned one-dimensional arrays.")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("Warning labels must contain only zero and one.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Warning probabilities must be finite and within [0,1].")
    return truth.astype(np.int8), scores


def classification_metrics(y_true, probability, threshold: float) -> dict:
    """Row-weighted diagnostics; zero-denominator precision/recall are zero.

    AP uses sklearn's stepwise average precision. A fold without positives has
    AP=0 by the same sklearn convention, marked explicitly in ``pr_auc_status``.
    The caller must state whether rows are terminal OOF, terminal test or all
    test rows; those three populations are not interchangeable.
    """
    truth, scores = _binary_arrays(y_true, probability)
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The warning threshold must be within [0,1].")
    predicted = scores >= threshold
    actual = truth.astype(bool)
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    tn = int(np.sum(~predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * tp / (5 * tp + 4 * fn + fp) if 5 * tp + 4 * fn + fp else 0.0
    positives = int(actual.sum())
    ap = float(average_precision_score(truth, scores)) if positives else 0.0
    return {
        "n": int(len(truth)), "positives": positives,
        "prevalence": positives / len(truth), "threshold": threshold,
        "pr_auc": ap, "pr_auc_status": "defined" if positives else "zero_no_positive_labels",
        "precision": float(precision), "recall": float(recall), "f2": float(f2),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "predicted_positives": tp + fp,
    }


def threshold_from_oof(y_true, probability) -> float:
    """Maximize F2 on train-only OOF probabilities; exact ties choose higher.

    Calibration reuses the OOF labels, so metrics at this threshold are selection
    diagnostics, not an unbiased estimate. The threshold convention is ``>=``.
    The extra threshold 1 includes a valid predict-none option unless a model
    assigns probability exactly 1; its confusion counts are computed explicitly.
    """
    truth, scores = _binary_arrays(y_true, probability)
    if not truth.any():
        return 1.0
    precision, recall, thresholds = precision_recall_curve(truth, scores)
    p, r = precision[:-1], recall[:-1]
    denominator = 4 * p + r
    f2 = np.divide(5 * p * r, denominator, out=np.zeros_like(p), where=denominator > 0)
    last_f2 = classification_metrics(truth, scores, 1.0)["f2"]
    candidate_thresholds = np.append(thresholds, 1.0)
    candidate_f2 = np.append(f2, last_f2)
    best = float(candidate_f2.max())
    # Floating-point PR calculations can differ at the final bit for an exact
    # rational-score tie. This tolerance only resolves numerical equality.
    tied = np.isclose(candidate_f2, best, rtol=1e-12, atol=1e-15)
    return float(candidate_thresholds[tied].max())


@dataclass
class WarningHead:
    horizon: int
    feature_names: list[str]
    estimator: LogisticRegression

    def predict_probability(self, features_df: pd.DataFrame) -> np.ndarray:
        observed = feature_columns(features_df)
        if observed != self.feature_names:
            raise ValueError("Warning feature columns/order differ from the training schema.")
        matrix = features_df[self.feature_names].to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("Warning inputs contain nonfinite feature values.")
        class_one = int(np.flatnonzero(self.estimator.classes_ == 1)[0])
        return self.estimator.predict_proba(features_df[self.feature_names])[:, class_one]

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        probability = self.predict_probability(features_df)
        return np.column_stack([1 - probability, probability])

    def predict(self, features_df: pd.DataFrame, threshold: float) -> np.ndarray:
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("The warning threshold must be within [0,1].")
        return (self.predict_probability(features_df) >= threshold).astype(np.int8)


def fit_warning(features_df: pd.DataFrame, raw_targets, N: int) -> WarningHead:
    """Fit the fixed class-weighted LogisticRegression family; no N selection.

    ``raw_targets`` must be uncapped training RUL, even when the RUL regression
    candidate uses capped labels. Per-engine row weights and class balancing
    are both applied, as frozen in protocol_v3.md.
    """
    if N not in HORIZONS:
        raise ValueError(f"Warning horizon must be one of {HORIZONS}.")
    names = feature_columns(features_df)
    targets = np.asarray(raw_targets, dtype=float)
    if targets.ndim != 1 or len(targets) != len(features_df) or not len(targets):
        raise ValueError("Uncapped warning RUL labels must align with training rows.")
    if not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("Uncapped warning RUL labels must be finite and nonnegative.")
    if not np.isfinite(features_df[names].to_numpy(dtype=float)).all():
        raise ValueError("Warning training features contain nonfinite values.")
    labels = (targets <= N).astype(np.int8)
    if np.unique(labels).size != 2:
        raise ValueError("Warning fitting requires both label classes; no silent fallback model.")
    estimator = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    estimator.fit(features_df[names], labels, sample_weight=engine_weights(features_df))
    return WarningHead(horizon=int(N), feature_names=names, estimator=estimator)


def select_warning_horizon(oof_by_horizon: Mapping[int, Mapping]) -> dict:
    """Select N by pooled OOF max-F2, then recall, then the smaller horizon.

    Each value accepts keys ``y_true`` and ``probability``. All rows must already
    be TRAIN-only terminal-like OOF points from the same 15 frozen folds.
    """
    if set(oof_by_horizon) != set(HORIZONS):
        raise ValueError(f"OOF predictions are required for exactly horizons {HORIZONS}.")
    candidates = []
    row_count = None
    for horizon in HORIZONS:
        values = oof_by_horizon[horizon]
        truth, probability = _binary_arrays(values["y_true"], values["probability"])
        if row_count is not None and len(truth) != row_count:
            raise ValueError("All horizons must use the same number of OOF rows.")
        row_count = len(truth)
        threshold = threshold_from_oof(truth, probability)
        candidate = {"N": horizon, **classification_metrics(truth, probability, threshold)}
        candidates.append(candidate)
    selected = min(candidates, key=lambda item: (-item["f2"], -item["recall"], item["N"]))
    return {"N": selected["N"], "threshold": selected["threshold"],
            "metrics": selected, "candidates": candidates,
            "selection_note": "Train-OOF threshold/horizon selection diagnostics; optimistic, not unbiased validation."}


# Readable aliases for callers; all use the same exact row/threshold convention.
warning_metrics = classification_metrics
evaluate_warning = classification_metrics
