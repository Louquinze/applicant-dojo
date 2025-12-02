"""
Utilities for analyzing linear relationships between sensor time series.

This module focuses on:
- Pairwise linear correlation between sensors
- Optional lag search to detect delayed relationships

You can use it either as a library or as a small CLI script:

    python -m src.relationships --max-lag 10
"""

from __future__ import annotations

import argparse
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_simulator import IndustrialDataSimulator
from .data_processing import ingest_data


def _prepare_wide_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-format sensor data into a wide time-indexed DataFrame.

    Index: timestamp
    Columns: one column per sensor name
    Values: sensor 'value'
    """
    df = data.copy()
    df = df.sort_values("timestamp")
    df = df.pivot_table(
        index="timestamp",
        columns="sensor",
        values="value",
        aggfunc="mean",
    )
    # Ensure consistent time ordering
    df = df.sort_index()
    return df


def _lagged_corr(
    x: pd.Series,
    y: pd.Series,
    max_lag: int,
) -> Tuple[int, float]:
    """
    Compute the maximum absolute Pearson correlation between x and y
    over integer lags in range [-max_lag, max_lag].

    Positive lag means y is shifted *forward* (y responds later).
    Returns (best_lag, best_corr).
    """
    best_lag = 0
    best_corr = np.nan

    # Align on common index to simplify shifting logic
    x = x.astype(float)
    y = y.astype(float)
    common_idx = x.index.intersection(y.index)
    x = x.loc[common_idx]
    y = y.loc[common_idx]

    if x.empty or y.empty:
        return best_lag, np.nan

    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            x_lagged = x
            y_lagged = y
        elif lag > 0:
            # y occurs later than x
            x_lagged = x.iloc[:-lag]
            y_lagged = y.iloc[lag:]
        else:
            # x occurs later than y
            k = -lag
            x_lagged = x.iloc[k:]
            y_lagged = y.iloc[:-k]

        if len(x_lagged) < 3 or len(y_lagged) < 3:
            continue

        valid_mask = (~x_lagged.isna()) & (~y_lagged.isna())
        if not valid_mask.any():
            continue

        corr = np.corrcoef(
            x_lagged[valid_mask],
            y_lagged[valid_mask],
        )[0, 1]

        if np.isnan(corr):
            continue

        if np.isnan(best_corr) or abs(corr) > abs(best_corr):
            best_corr = float(corr)
            best_lag = lag

    return best_lag, best_corr


def analyze_sensor_relationships(
    data: pd.DataFrame,
    max_lag: int = 10,
    sensors: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Analyze linear relationships between sensor time series.

    For each sensor pair, compute:
    - best_lag: integer lag (in number of samples) that maximizes absolute correlation
    - best_corr: Pearson correlation at that lag

    Args:
        data: DataFrame from ingest_data()
        max_lag: Maximum lag (in samples) to search in both directions
        sensors: Optional subset of sensors to analyze (defaults to all)

    Returns:
        DataFrame with columns:
            - sensor_x
            - sensor_y
            - best_lag
            - best_corr
    """
    wide = _prepare_wide_dataframe(data)
    available_sensors = list(wide.columns)

    if sensors is None:
        sensors = available_sensors
    else:
        sensors = [s for s in sensors if s in available_sensors]

    results: List[Dict[str, object]] = []
    for s1, s2 in combinations(sensors, 2):
        lag, corr = _lagged_corr(wide[s1], wide[s2], max_lag=max_lag)
        results.append(
            {
                "sensor_x": s1,
                "sensor_y": s2,
                "best_lag": lag,
                "best_corr": corr,
            }
        )

    return pd.DataFrame(results)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze linear relationships between simulated sensor time series.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=10,
        help="Maximum lag (in samples) to search in both directions.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="Number of batches to simulate.",
    )
    parser.add_argument(
        "--batch-duration",
        type=int,
        default=60,
        help="Duration (seconds) of each batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the simulator.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    sim = IndustrialDataSimulator(seed=args.seed)
    batches = sim.get_batch_readings(
        num_batches=args.num_batches,
        batch_duration=args.batch_duration,
    )
    if not batches:
        raise RuntimeError("Simulator returned no successful batches for analysis.")

    clean = ingest_data(batches, validate=True)
    rel = analyze_sensor_relationships(clean, max_lag=args.max_lag)

    if rel.empty:
        print("No sensor relationships could be computed.")
        return

    # Sort by absolute correlation strength
    rel = rel.sort_values("best_corr", key=lambda s: s.abs(), ascending=False)

    print("Pairwise sensor relationships (max-abs Pearson correlation with lag):")
    for _, row in rel.iterrows():
        print(
            f"- {row['sensor_x']} ↔ {row['sensor_y']}: "
            f"corr={row['best_corr']:.3f} at lag={int(row['best_lag'])} samples"
        )


if __name__ == "__main__":
    main()


