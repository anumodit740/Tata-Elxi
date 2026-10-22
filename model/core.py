"""Causal, train-fitted primitives for the C-MAPSS demonstration.

No engine identifier, cycle index, future sensor reading, or eventual trajectory
length is a model feature. Training labels are uncapped cycles-to-failure. Sensor
cleaning and normalization are fitted on training engines only. First 20 cycles
are an explicitly assumed nominal anomaly reference, not certified fault labels.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ID_COLUMNS = ["engine_id", "cycle"]
SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
VALUE_COLUMNS = SETTING_COLUMNS + SENSOR_COLUMNS
CANONICAL_COLUMNS = ID_COLUMNS + VALUE_COLUMNS


class DataValidationError(ValueError):
    """Input cannot be used safely without an explicit data-quality exception."""


def validate_trajectory(df: pd.DataFrame, allow_missing: bool = False) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise DataValidationError("A nonempty trajectory DataFrame is required.")
    if list(df.columns) != CANONICAL_COLUMNS:
        raise DataValidationError("Expected exactly 26 canonical columns in their documented order.")
    for col in CANONICAL_COLUMNS:
        if (not pd.api.types.is_numeric_dtype(df[col])
                or pd.api.types.is_bool_dtype(df[col])
                or pd.api.types.is_complex_dtype(df[col])):
            raise DataValidationError(f"Non-real-numeric column: {col}")
    values = df.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise DataValidationError("Infinite input values are invalid.")
    if not allow_missing and np.isnan(values).any():
        raise DataValidationError("Missing inference values require human review; no automatic work order.")
    identifiers = values[:, :2]
    if (not np.isfinite(identifiers).all() or (identifiers <= 0).any()
            or not np.equal(identifiers, np.floor(identifiers)).all()):
        raise DataValidationError("Engine identifiers and cycles must be positive finite integers.")
    if df.duplicated(ID_COLUMNS).any():
        raise DataValidationError("Duplicate engine/cycle pairs are invalid.")
    for _, group in df.groupby("engine_id", sort=False):
        if np.any(np.diff(group["cycle"].to_numpy()) <= 0):
            raise DataValidationError("Cycles must increase within each engine; never silently reorder events.")


def load_trajectory_file(path: str | Path) -> pd.DataFrame:
    """Read the 26-column NASA format; permit NaN only for explicit training cleaning."""
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None)
    except Exception as exc:
        raise DataValidationError(f"Unable to read trajectory file: {path}") from exc
    if df.shape[1] != len(CANONICAL_COLUMNS):
        raise DataValidationError(f"Expected 26 columns; found {df.shape[1]} in {path}.")
    df.columns = CANONICAL_COLUMNS
    for col in CANONICAL_COLUMNS:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except Exception as exc:
            raise DataValidationError(f"Invalid numeric input in {col}.") from exc
    validate_trajectory(df, allow_missing=True)
    df[ID_COLUMNS] = df[ID_COLUMNS].astype("int64")
    return df


def fit_cleaner(train_df: pd.DataFrame) -> dict:
    validate_trajectory(train_df, allow_missing=True)
    medians = train_df[VALUE_COLUMNS].median()
    if medians.isna().any():
        raise DataValidationError("An entirely missing training column has no supported fallback.")
    return {"fallback_medians": {k: float(v) for k, v in medians.items()},
            "policy": "past-only within-engine forward fill; training-only median fallback"}


def clean_training(df: pd.DataFrame, cleaner: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_trajectory(df, allow_missing=True)
    out = df.copy()
    missing = out[VALUE_COLUMNS].isna()
    filled = out.groupby("engine_id", sort=False)[VALUE_COLUMNS].ffill()
    out[VALUE_COLUMNS] = filled.fillna(pd.Series(cleaner["fallback_medians"])).astype(float)
    validate_trajectory(out)
    return out, missing


def past_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Current values, trailing mean, endpoint slope; no cross-engine or future access.

    Short histories use available observations. Slopes divide by elapsed cycle
    count rather than observation count, preserving meaning across missing cycles.
    """
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    validate_trajectory(df)
    values = df[VALUE_COLUMNS].to_numpy(dtype=float)
    means = np.empty_like(values)
    trends = np.empty_like(values)
    for positions in df.groupby("engine_id", sort=False).indices.values():
        x = values[positions]
        cycles = df.iloc[positions]["cycle"].to_numpy(dtype=float)
        end = np.arange(len(x))
        start = np.maximum(0, end - window + 1)
        prefix = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
        means[positions] = (prefix[end + 1] - prefix[start]) / (end - start + 1)[:, None]
        elapsed = cycles - cycles[start]
        trends[positions] = np.divide(x - x[start], elapsed[:, None],
                                     out=np.zeros_like(x), where=elapsed[:, None] > 0)
    out = df[ID_COLUMNS].reset_index(drop=True).copy()
    for matrix, suffix in [(values, ""), (means, f"_mean_{window}"), (trends, f"_trend_{window}")]:
        out = pd.concat([out, pd.DataFrame(matrix, columns=[c + suffix for c in VALUE_COLUMNS])], axis=1)
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise DataValidationError("Feature computation produced nonfinite values.")
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    if df.columns.duplicated().any() or list(df.columns[:2]) != ID_COLUMNS:
        raise DataValidationError("Feature frames must start with engine_id,cycle and contain unique columns.")
    cols = [c for c in df.columns if c not in ID_COLUMNS]
    if not cols:
        raise DataValidationError("No model features remain.")
    return cols


def rul_targets(raw_df: pd.DataFrame) -> np.ndarray:
    validate_trajectory(raw_df, allow_missing=True)
    return (raw_df.groupby("engine_id")["cycle"].transform("max") - raw_df["cycle"]).to_numpy(dtype=float)


def engine_weights(features: pd.DataFrame) -> np.ndarray:
    counts = features.groupby("engine_id")["engine_id"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    return weights * len(weights) / weights.sum()


class TrajectoryPreprocessor:
    """Train-only cleaning, optional regime normalization, variance filter and scaling.

    FD002/FD004 use six KMeans regimes fitted on standardized operating settings.
    Regime-specific sensor means/scales are estimated on training engines only.
    Constant features are selected by observed training variance, never by a
    hard-coded sensor list copied from a publication or a test-file inspection.
    """
    def __init__(self, subset: str, window: int = 10, random_state: int = 42):
        self.subset, self.window, self.random_state = subset, window, random_state

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = raw.copy()
        if self.regime_model is not None:
            settings = self.setting_scaler.transform(raw[SETTING_COLUMNS])
            groups = self.regime_model.predict(settings)
            x = raw[SENSOR_COLUMNS].to_numpy(dtype=float)
            for regime in range(6):
                mask = groups == regime
                x[mask] = (x[mask] - self.regime_means[regime]) / self.regime_scales[regime]
            out[SENSOR_COLUMNS] = x
        return out

    def fit_transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        self.cleaner = fit_cleaner(raw)
        clean, missing = clean_training(raw, self.cleaner)
        self.training_missing_counts = {k: int(v) for k, v in missing.sum().items()}
        self.regime_model = None
        self.regime_counts = {}
        self.raw_variances = {k: float(v) for k, v in clean[VALUE_COLUMNS].var(ddof=0).items()}
        self.dropped_raw_columns = [c for c, v in self.raw_variances.items() if v <= 1e-12]
        if self.subset in {"FD002", "FD004"}:
            self.setting_scaler = StandardScaler()
            settings = self.setting_scaler.fit_transform(clean[SETTING_COLUMNS])
            self.regime_model = KMeans(n_clusters=6, n_init=10, random_state=self.random_state)
            groups = self.regime_model.fit_predict(settings)
            sensors = clean[SENSOR_COLUMNS].to_numpy(dtype=float)
            means, scales = [], []
            global_scale = sensors.std(axis=0)
            global_scale[global_scale <= 1e-12] = 1.0
            for regime in range(6):
                rows = sensors[groups == regime]
                self.regime_counts[str(regime)] = int(len(rows))
                means.append(rows.mean(axis=0))
                std = rows.std(axis=0)
                scales.append(np.where(std > 1e-12, std, global_scale))
            self.regime_means, self.regime_scales = np.asarray(means), np.asarray(scales)
        frame = past_features(self._normalize(clean), self.window)
        self.all_feature_names = feature_columns(frame)
        # Remove raw constants and all their derived features before variance filtering.
        self.candidate_feature_names = [c for c in self.all_feature_names
                                        if not any(c == raw_col or c.startswith(raw_col + "_")
                                                   for raw_col in self.dropped_raw_columns)]
        self.variance_selector = VarianceThreshold(threshold=1e-12)
        x = self.variance_selector.fit_transform(frame[self.candidate_feature_names])
        support = self.variance_selector.get_support()
        self.feature_names = [c for c, keep in zip(self.candidate_feature_names, support) if keep]
        self.dropped_feature_columns = [c for c in self.all_feature_names if c not in self.feature_names]
        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(x)
        return pd.concat([frame[ID_COLUMNS], pd.DataFrame(x, columns=self.feature_names)], axis=1)

    def transform(self, raw: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
        validate_trajectory(raw, allow_missing=not strict)
        clean, _ = clean_training(raw, self.cleaner)
        frame = past_features(self._normalize(clean), self.window)
        x = self.variance_selector.transform(frame[self.candidate_feature_names])
        x = self.scaler.transform(x)
        return pd.concat([frame[ID_COLUMNS], pd.DataFrame(x, columns=self.feature_names)], axis=1)

    def describe(self) -> dict:
        return {"window": self.window, "feature_names": self.feature_names,
                "dropped_raw_columns": self.dropped_raw_columns,
                "dropped_feature_columns": self.dropped_feature_columns,
                "training_missing_counts": self.training_missing_counts,
                "normalizer_regime_counts": self.regime_counts,
                "normalization": "six train-fitted operating regimes" if self.regime_model is not None else "single regime; train-fitted feature scaler",
                "cleaner": self.cleaner,
                "feature_scaler_mean": self.scaler.mean_.tolist(),
                "feature_scaler_scale": self.scaler.scale_.tolist()}


class MedianRUL:
    def fit(self, features: pd.DataFrame, targets: np.ndarray):
        self.feature_names = feature_columns(features)
        self.median = float(np.median(targets))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.median)


class TabularRUL:
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, features: pd.DataFrame, targets: np.ndarray):
        self.feature_names = feature_columns(features)
        self.estimator.fit(features[self.feature_names], targets, sample_weight=engine_weights(features))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.maximum(0.0, self.estimator.predict(features[self.feature_names]))


class EchoStateRUL:
    """Lightweight causal recurrent reservoir with a trainable ridge readout.

    This is an Echo State Network, not an LSTM. Fixed seeded recurrent weights
    summarize past observations; state resets to zero for each engine. The
    readout uses current standardized features and 24 reservoir units by default.
    """
    def __init__(self, units: int = 24, alpha: float = 10.0, random_state: int = 42):
        self.units, self.alpha, self.random_state = units, alpha, random_state

    def _states(self, features: pd.DataFrame) -> np.ndarray:
        x = features[self.feature_names].to_numpy(dtype=float)
        result = np.empty((len(x), self.units), dtype=float)
        for positions in features.groupby("engine_id", sort=False).indices.values():
            state = np.zeros(self.units)
            for position in positions:
                state = 0.5 * state + 0.5 * np.tanh(x[position] @ self.input_weights
                                                    + state @ self.recurrent_weights + self.bias)
                result[position] = state
        return np.hstack([x, result])

    def fit(self, features: pd.DataFrame, targets: np.ndarray):
        self.feature_names = feature_columns(features)
        rng = np.random.default_rng(self.random_state)
        self.input_weights = rng.normal(0, 0.5 / np.sqrt(len(self.feature_names)),
                                        (len(self.feature_names), self.units))
        weights = rng.normal(0, 1, (self.units, self.units))
        radius = float(np.max(np.abs(np.linalg.eigvals(weights))))
        self.recurrent_weights = weights * (0.8 / radius)
        self.bias = rng.normal(0, 0.05, self.units)
        self.readout = Ridge(alpha=self.alpha)
        self.readout.fit(self._states(features), targets, sample_weight=engine_weights(features))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.maximum(0.0, self.readout.predict(self._states(features)))


def fit_anomaly(raw_df: pd.DataFrame, features_df: pd.DataFrame) -> dict:
    if len(raw_df) != len(features_df) or not np.array_equal(raw_df[ID_COLUMNS].to_numpy(), features_df[ID_COLUMNS].to_numpy()):
        raise DataValidationError("Raw and feature rows must align for nominal-reference fitting.")
    # Operating settings describe demand, not degradation. Quantized settings
    # also have zero MAD and can overwhelm a sensor-health distance.
    names = [name for name in feature_columns(features_df) if name.startswith("sensor_")]
    reference_mask = raw_df["cycle"].to_numpy() <= 20
    if not reference_mask.any():
        raise DataValidationError("No first-20-cycle nominal reference is available.")
    reference = features_df.loc[reference_mask, names].to_numpy(dtype=float)
    median = np.median(reference, axis=0)
    mad = np.median(np.abs(reference - median), axis=0)
    std = np.std(reference, axis=0)
    retained = std > 1e-8
    if not retained.any():
        raise DataValidationError("No variable sensor features in nominal reference.")
    dropped = [name for name, keep in zip(names, retained) if not keep]
    names = [name for name, keep in zip(names, retained) if keep]
    reference, median, mad, std = reference[:, retained], median[retained], mad[retained], std[retained]
    robust_scale = 1.4826 * mad
    fallback = robust_scale <= 1e-8
    scale = np.where(fallback, std, robust_scale)
    scores = np.mean(np.abs((reference - median) / scale), axis=1)
    return {"feature_names": names, "median": median.tolist(), "scale": scale.tolist(),
            "threshold": float(np.quantile(scores, 0.99)), "quantile": 0.99,
            "reference_rows": int(reference_mask.sum()),
            "zero_mad_std_fallback_features": [name for name, used in zip(names, fallback) if used],
            "nominal_constant_features_dropped": dropped,
            "assumption": "cycles1-20 assumed nominal; no true anomaly onset labels supplied",
            "score_definition": "sensor-only mean absolute nominal-reference distance; MAD scale, nominal std fallback for zero MAD"}


def score_anomaly(model: dict, features_df: pd.DataFrame) -> np.ndarray:
    x = features_df[model["feature_names"]].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise DataValidationError("Anomaly inference features must be finite.")
    return np.mean(np.abs((x - np.asarray(model["median"])) / np.asarray(model["scale"])), axis=1)


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or truth.size == 0 or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric inputs must be matching nonempty finite vectors.")
    error = prediction - truth
    exponent = np.where(error < 0, -error / 13.0, error / 10.0)
    if np.max(exponent) > 700:
        raise ValueError("NASA score would overflow; inspect unreasonable model predictions.")
    return {"rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "nasa_score": float(np.expm1(exponent).sum()),
            "nasa_score_mean": float(np.expm1(exponent).mean()),
            "overprediction_fraction": float(np.mean(error > 0)), "n_engines": int(len(error))}
