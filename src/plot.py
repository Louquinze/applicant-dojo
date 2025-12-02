"""
Simple plotting utility for visualizing simulated industrial sensor data.

Usage (from project root):

    python -m src.plot --sensor temperature --method zscore --threshold 3.0

This will:
1. Generate a few batches of simulated data
2. Ingest and clean them using `ingest_data`
3. Run anomaly detection with `detect_anomalies`
4. Plot the selected sensor's time series, highlighting anomalies

Note: This script requires `matplotlib` to be installed:

    pip install matplotlib
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .data_simulator import IndustrialDataSimulator
from .data_processing import ingest_data, detect_anomalies, summarize_metrics


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot simulated industrial sensor data.")
    parser.add_argument(
        "--sensor",
        type=str,
        default="temperature",
        help="Sensor name to visualize (e.g. 'temperature', 'pressure').",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="zscore",
        choices=["zscore", "iqr", "rolling"],
        help="Anomaly detection method.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Threshold parameter for the selected anomaly method.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=3,
        help="Number of batches to simulate.",
    )
    parser.add_argument(
        "--batch-duration",
        type=int,
        default=60,
        help="Duration (in seconds) of each simulated batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the simulator.",
    )
    parser.add_argument(
        "--view",
        type=str,
        default="timeseries",
        choices=["timeseries", "metrics"],
        help="What to visualize: raw time series or aggregated metrics.",
    )
    return parser


def _plot_sensor_time_series(
    data: pd.DataFrame,
    sensor_name: str,
    title: Optional[str] = None,
) -> None:
    """Plot a single sensor's time series, highlighting anomalies if present."""
    sensor_df = data[data["sensor"] == sensor_name].copy()
    if sensor_df.empty:
        raise ValueError(f"No data available for sensor '{sensor_name}' to plot")

    sensor_df = sensor_df.sort_values("timestamp")

    has_anomalies = "is_anomaly" in sensor_df.columns and sensor_df["is_anomaly"].any()

    # Use a pleasant seaborn style
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    # Base line for all readings
    sns.lineplot(
        data=sensor_df,
        x="timestamp",
        y="value",
        ax=ax,
        label=f"{sensor_name} value",
        color="tab:blue",
        alpha=0.8,
    )

    if has_anomalies:
        anomalies = sensor_df[sensor_df["is_anomaly"]]
        sns.scatterplot(
            data=anomalies,
            x="timestamp",
            y="value",
            ax=ax,
            color="tab:red",
            label="Anomalies",
            zorder=3,
            s=40,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(f"Value ({sensor_df['unit'].iloc[0]})")
    ax.legend()

    if title is None:
        method = sensor_df.get("detection_method")
        method_name = method.iloc[0] if method is not None and not method.isna().all() else "none"
        title = f"Sensor '{sensor_name}' time series (method={method_name})"

    ax.set_title(title)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def _plot_summary_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """Visualize summarize_metrics() output using seaborn."""
    if not metrics:
        raise ValueError("No metrics provided for plotting")

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index").reset_index()
    metrics_df = metrics_df.rename(columns={"index": "sensor"})

    sns.set_theme(style="whitegrid")

    # 1) Mean & spread per sensor
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=metrics_df,
        x="sensor",
        y="mean",
        yerr=metrics_df["std"],
        ax=ax,
        color="tab:blue",
        alpha=0.8,
    )
    ax.set_title("Sensor mean values with spread (std)")
    ax.set_ylabel("Mean value")
    plt.tight_layout()

    # 2) Quality & anomaly rates per sensor
    rate_cols = ["good_quality_pct", "anomaly_rate"]
    rate_df = metrics_df[["sensor"] + rate_cols].melt(
        id_vars="sensor", var_name="metric", value_name="value"
    )

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=rate_df,
        x="sensor",
        y="value",
        hue="metric",
        ax=ax2,
    )
    ax2.set_title("Data quality and anomaly rates per sensor")
    ax2.set_ylabel("Value (percent or fraction)")
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    sim = IndustrialDataSimulator(seed=args.seed)

    # Step 1: simulate data batches (some may fail)
    batches = sim.get_batch_readings(
        num_batches=args.num_batches,
        batch_duration=args.batch_duration,
    )

    if not batches:
        raise RuntimeError("Simulator returned no successful batches to plot.")

    # Step 2: ingest and clean
    clean_data = ingest_data(batches, validate=True)

    # Step 3: run anomaly detection
    anomaly_data = detect_anomalies(
        clean_data,
        sensor_name=args.sensor,
        method=args.method,
        threshold=args.threshold,
    )

    if args.view == "timeseries":
        # Plot selected sensor time series
        _plot_sensor_time_series(anomaly_data, sensor_name=args.sensor)
    else:
        # Plot summarize_metrics() output
        metrics = summarize_metrics(anomaly_data, group_by="sensor")
        _plot_summary_metrics(metrics)


if __name__ == "__main__":
    main()


