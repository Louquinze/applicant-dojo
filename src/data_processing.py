"""
Core Data Processing Functions for FDSE Challenge

CANDIDATE TASK: Implement the three functions below according to their specifications.

These functions form the core of an industrial data processing pipeline.
You will work with real-world challenges like missing data, connection failures,
and noisy sensor readings.

IMPORTANT NOTES:
- Function signatures (names, parameters, return types) must not be changed
- You may add helper functions in this file or create new modules
- Focus on robustness, error handling, and data quality
- Document your assumptions and trade-offs in NOTES.md
- Aim for production-quality code, not just passing tests
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def ingest_data(
    data_batches: List[pd.DataFrame],
    validate: bool = True,
) -> pd.DataFrame:
    """
    Ingest and consolidate multiple batches of industrial sensor data.
    
    This function must handle real-world data quality issues:
    - Missing or null values
    - Duplicate readings
    - Out-of-order timestamps
    - Data from different sensors with different units
    - Potentially empty batches
    """
    if len(data_batches) == 0:
        raise ValueError("data_batches must not be empty")

    # Filter out obviously invalid or empty batches while validating structure
    valid_batches: List[pd.DataFrame] = []
    required_columns = {"timestamp", "sensor", "value", "unit", "quality"}

    for idx, batch in enumerate(data_batches):
        if batch is None:
            # Skip missing batches
            continue
        if not isinstance(batch, pd.DataFrame):
            raise ValueError(f"Batch at index {idx} is not a pandas DataFrame")
        if batch.empty:
            # Ignore empty batches
            continue

        # Ensure expected schema; this keeps downstream logic predictable
        if not required_columns.issubset(set(batch.columns)):
            raise ValueError(
                f"Batch at index {idx} is missing required columns: "
                f"{required_columns - set(batch.columns)}"
            )

        # Work on a copy to avoid mutating caller-owned DataFrames
        valid_batches.append(batch.copy())

    if not valid_batches:
        # All batches were empty/invalid
        raise ValueError("No valid, non-empty data batches provided")

    # Concatenate all valid batches
    data = pd.concat(valid_batches, ignore_index=True)

    if not validate:
        # Even without full validation, maintain chronological order when possible
        if "timestamp" in data.columns:
            data = data.sort_values("timestamp").reset_index(drop=True)
        return data

    # --- Data Cleaning & Normalization ---

    # Coerce timestamps to pandas datetime and drop rows where this fails
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"])

    # Remove exact duplicate records; this is robust and keeps one canonical copy
    data = data.drop_duplicates()

    # We intentionally keep NaN values in "value" for later processing

    # Optional: normalize quality flag casing for consistency
    if "quality" in data.columns:
        data["quality"] = data["quality"].astype(str).str.upper()

    # Final ordering by timestamp to ensure a clean time series
    data = data.sort_values("timestamp").reset_index(drop=True)

    return data


def detect_anomalies(
    data: pd.DataFrame,
    sensor_name: str,
    method: str = "zscore",
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Detect anomalies in sensor data using statistical methods.
    """
    if "sensor" not in data.columns:
        raise ValueError("Input data must contain a 'sensor' column")
    if sensor_name not in data["sensor"].unique():
        raise ValueError(f"Sensor '{sensor_name}' not found in data")

    supported_methods = {"zscore", "iqr", "rolling"}
    if method not in supported_methods:
        raise ValueError(
            f"Unsupported method '{method}'. Supported methods: {sorted(supported_methods)}"
        )

    result = data.copy()

    # Initialize output columns
    result["is_anomaly"] = False
    result["anomaly_score"] = np.nan
    result["detection_method"] = None

    # Focus on the target sensor
    sensor_mask = result["sensor"] == sensor_name
    sensor_data = result.loc[sensor_mask]

    if sensor_data.empty:
        raise ValueError(f"No data available for sensor '{sensor_name}'")

    value_series = sensor_data["value"].astype(float)
    valid_values = value_series.dropna()

    if len(valid_values) < 2:
        raise ValueError(
            f"Insufficient non-null data for sensor '{sensor_name}' "
            "to perform anomaly detection"
        )

    if method == "zscore":
        mean = valid_values.mean()
        std = valid_values.std(ddof=0)

        if std == 0 or np.isclose(std, 0.0):
            scores = (value_series - mean) * 0.0
        else:
            scores = (value_series - mean) / std

        is_anom = scores.abs() > threshold

    elif method == "iqr":
        q1 = valid_values.quantile(0.25)
        q3 = valid_values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or np.isclose(iqr, 0.0):
            scores = pd.Series(0.0, index=value_series.index)
        else:
            # Scaled distance from nearest quartile
            lower_diff = (q1 - value_series).clip(lower=0)
            upper_diff = (value_series - q3).clip(lower=0)
            raw_dist = lower_diff + upper_diff
            scores = raw_dist / (iqr * threshold)

        is_anom = scores.abs() > 1.0

    elif method == "rolling":
        # Simple rolling z-score style detector
        window = max(int(threshold), 3)
        rolling_mean = valid_values.rolling(window=window, min_periods=2).mean()
        rolling_std = valid_values.rolling(window=window, min_periods=2).std(ddof=0)

        scores = pd.Series(index=value_series.index, dtype=float)
        aligned_mean = rolling_mean.reindex(value_series.index)
        aligned_std = rolling_std.reindex(value_series.index)

        with np.errstate(divide="ignore", invalid="ignore"):
            scores = (value_series - aligned_mean) / aligned_std

        scores = scores.replace([np.inf, -np.inf], np.nan)
        is_anom = scores.abs() > threshold

    # Apply results for the target sensor
    is_anom = is_anom.fillna(False)
    result.loc[sensor_mask, "anomaly_score"] = scores
    result.loc[sensor_mask, "is_anomaly"] = is_anom.astype(bool)
    result.loc[sensor_mask, "detection_method"] = method

    result["is_anomaly"] = result["is_anomaly"].astype(bool)
    result["anomaly_score"] = result["anomaly_score"].astype(float)

    return result


def summarize_metrics(
    data: pd.DataFrame,
    group_by: Optional[str] = "sensor",
    time_window: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Generate summary statistics for industrial sensor data.
    """
    if data.empty:
        raise ValueError("Cannot summarize metrics for an empty DataFrame")

    if group_by is None:
        raise ValueError("group_by must be specified for summarization")

    if group_by not in data.columns:
        raise ValueError(f"group_by column '{group_by}' does not exist in data")

    def _compute_group_metrics(df: pd.DataFrame) -> Dict[str, float]:
        values = df["value"].astype(float)

        mean = float(values.mean(skipna=True))
        std = float(values.std(ddof=0, skipna=True))
        min_val = float(values.min(skipna=True)) if not values.dropna().empty else np.nan
        max_val = float(values.max(skipna=True)) if not values.dropna().empty else np.nan

        count = int(len(df))
        null_count = int(values.isna().sum())

        # Data quality metrics
        if "quality" in df.columns:
            quality_series = df["quality"].astype(str).str.upper()
            total_quality = (quality_series.notna()).sum()
            good_count = (quality_series == "GOOD").sum()
            good_quality_pct = float(
                (good_count / total_quality) * 100.0 if total_quality > 0 else 0.0
            )
        else:
            good_quality_pct = float("nan")

        # Anomaly stats if available
        if "is_anomaly" in df.columns:
            is_anom = df["is_anomaly"].astype(bool)
            anomaly_rate = float(is_anom.mean()) if len(is_anom) > 0 else 0.0
        else:
            anomaly_rate = float("nan")

        return {
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "count": count,
            "null_count": null_count,
            "good_quality_pct": good_quality_pct,
            "anomaly_rate": anomaly_rate,
        }

    metrics: Dict[str, Dict[str, float]] = {}

    if time_window is not None:
        if "timestamp" not in data.columns:
            raise ValueError(
                "time_window specified but 'timestamp' column is missing from data"
            )

        df = data.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        grouped = df.groupby(
            [pd.Grouper(key="timestamp", freq=time_window), group_by],
            dropna=False,
        )

        for (time_key, group_key), group_df in grouped:
            if group_df.empty:
                continue
            time_label = "unknown" if pd.isna(time_key) else str(time_key)
            composite_key = f"{group_key}@{time_label}"
            metrics[composite_key] = _compute_group_metrics(group_df)

        return metrics

    grouped = data.groupby(group_by, dropna=False)

    for group_value, group_df in grouped:
        if group_df.empty:
            continue
        key = str(group_value)
        metrics[key] = _compute_group_metrics(group_df)

    return metrics
