"""
Interactive dashboard for exploring simulated industrial sensor data.

Run from the project root:

    python -m src.dashboard

This will start a Dash app where you can:
- Generate new simulated data batches
- Choose a sensor and anomaly detection method
- Inspect the time series with anomalies highlighted
- View summary metrics per sensor
"""

from __future__ import annotations

from typing import List, Dict

import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_simulator import IndustrialDataSimulator
from .data_processing import ingest_data, detect_anomalies, summarize_metrics
from .relationships import analyze_sensor_relationships


def _create_app() -> Dash:
    app = Dash(__name__)
    app.title = "Industrial Sensor Dashboard"

    # Fixed color mapping per sensor for visual consistency
    sensor_colors: Dict[str, str] = {
        "temperature": "#1f77b4",  # blue
        "pressure": "#2ca02c",  # green
        "vibration": "#ff7f0e",  # orange
        "flow_rate": "#d62728",  # red
    }

    simulator = IndustrialDataSimulator(seed=42)

    app.layout = html.Div(
        style={"fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, sans-serif", "padding": "16px"},
        children=[
            html.H2("Industrial Sensor Dashboard"),
            html.Div(
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                children=[
                    html.Div(
                        style={"minWidth": "260px"},
                        children=[
                            html.H4("Controls"),
                            html.Label("Sensors"),
                            dcc.Checklist(
                                id="sensor-checklist",
                                options=[
                                    {"label": "Temperature", "value": "temperature"},
                                    {"label": "Pressure", "value": "pressure"},
                                    {"label": "Vibration", "value": "vibration"},
                                    {"label": "Flow rate", "value": "flow_rate"},
                                ],
                                value=["temperature", "pressure", "vibration", "flow_rate"],
                                style={"marginTop": "4px"},
                                labelStyle={"display": "block", "marginBottom": "2px"},
                            ),
                            html.Label("Anomaly method", style={"marginTop": "8px"}),
                            dcc.Dropdown(
                                id="method-dropdown",
                                options=[
                                    {"label": "Z-score", "value": "zscore"},
                                    {"label": "IQR", "value": "iqr"},
                                    {"label": "Rolling", "value": "rolling"},
                                ],
                                value="zscore",
                                clearable=False,
                            ),
                            html.Label("Threshold", style={"marginTop": "8px"}),
                            dcc.Slider(
                                id="threshold-slider",
                                min=1.0,
                                max=5.0,
                                step=0.5,
                                value=3.0,
                                marks={i: str(i) for i in range(1, 6)},
                            ),
                            html.Label("Number of batches", style={"marginTop": "8px"}),
                            dcc.Slider(
                                id="num-batches-slider",
                                min=1,
                                max=10,
                                step=1,
                                value=3,
                                marks={i: str(i) for i in range(1, 11)},
                            ),
                            html.Label("Batch duration (seconds)", style={"marginTop": "8px"}),
                            dcc.Slider(
                                id="batch-duration-slider",
                                min=10,
                                max=120,
                                step=10,
                                value=60,
                                marks={i: str(i) for i in range(10, 130, 20)},
                            ),
                            html.Button(
                                "Generate data",
                                id="generate-button",
                                n_clicks=0,
                                style={"marginTop": "12px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": 1, "minWidth": "300px"},
                        children=[
                            html.H4("Time Series"),
                            dcc.Graph(
                                id="timeseries-graph",
                                style={"height": "400px"},
                                config={
                                    "displayModeBar": True,
                                    "scrollZoom": True,
                                },
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                style={"marginTop": "100px"},
                children=[
                    html.H4("Summary Metrics"),
                    html.Div(
                        style={
                            "display": "flex",
                            "gap": "12px",
                            "flexWrap": "wrap",
                            "marginTop": "8px",
                        },
                        children=[
                            html.Div(
                                style={"flex": 1, "minWidth": "220px"},
                                children=[
                                    html.Label(
                                        "Good quality (%)", style={"fontSize": "12px"}
                                    ),
                                    dcc.Graph(
                                        id="metrics-good-graph",
                                        style={"height": "320px"},
                                        config={"displayModeBar": False},
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"flex": 1, "minWidth": "220px"},
                                children=[
                                    html.Label(
                                        "Anomaly rate (%)", style={"fontSize": "12px"}
                                    ),
                                    dcc.Graph(
                                        id="metrics-anomaly-graph",
                                        style={"height": "320px"},
                                        config={"displayModeBar": False},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                style={"marginTop": "150px"},
                children=[
                    html.H4("Sensor Relationships"),
                    html.Label("Max lag (samples)", style={"fontSize": "12px"}),
                    dcc.Slider(
                        id="max-lag-slider",
                        min=0,
                        max=20,
                        step=1,
                        value=10,
                        marks={0: "0", 5: "5", 10: "10", 15: "15", 20: "20"},
                    ),
                    dcc.Graph(
                        id="relationships-graph",
                        style={"height": "320px"},
                        config={"displayModeBar": False},
                    ),
                ],
            ),
            # Hidden div to store the latest ingested data as JSON
            dcc.Store(id="data-store"),
        ],
    )

    @app.callback(
        Output("data-store", "data"),
        Input("generate-button", "n_clicks"),
        Input("num-batches-slider", "value"),
        Input("batch-duration-slider", "value"),
        prevent_initial_call=False,
    )
    def generate_data(n_clicks: int | None, num_batches: int, batch_duration: int):
        batches = simulator.get_batch_readings(
            num_batches=num_batches,
            batch_duration=batch_duration,
        )
        if not batches:
            return None
        clean = ingest_data(batches, validate=True)
        # Store as JSON for round-tripping through Dash
        return clean.to_json(date_format="iso", orient="split")

    @app.callback(
        Output("timeseries-graph", "figure"),
        Output("metrics-good-graph", "figure"),
        Output("metrics-anomaly-graph", "figure"),
        Output("relationships-graph", "figure"),
        Input("data-store", "data"),
        Input("sensor-checklist", "value"),
        Input("method-dropdown", "value"),
        Input("threshold-slider", "value"),
        Input("max-lag-slider", "value"),
    )
    def update_plots(
        data_json: str | None,
        selected_sensors,
        method: str,
        threshold: float,
        max_lag: int,
    ):
        if not data_json:
            empty_fig = px.scatter()
            return empty_fig, empty_fig, empty_fig, empty_fig

        data = pd.read_json(data_json, orient="split")

        # Normalize selection to a list
        if isinstance(selected_sensors, str) or selected_sensors is None:
            selected_list = [selected_sensors] if selected_sensors else []
        else:
            selected_list = list(selected_sensors)

        if not selected_list:
            empty_fig = px.scatter()
            return empty_fig, empty_fig, empty_fig, empty_fig

        primary_sensor = selected_list[0]

        # Base data for plotting (no anomalies applied yet)
        visible_df = data[data["sensor"].isin(selected_list)].copy()
        visible_df = visible_df.sort_values("timestamp")

        if visible_df.empty:
            ts_fig = px.scatter(title="No data for selected sensors", height=400)
        else:
            # Compute per-sensor ranges and scale factors relative to primary sensor
            ranges: Dict[str, float] = {}
            for sensor in selected_list:
                vals = (
                    visible_df.loc[visible_df["sensor"] == sensor, "value"]
                    .astype(float)
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                if vals.empty:
                    ranges[sensor] = 0.0
                else:
                    ranges[sensor] = float(vals.max() - vals.min())

            primary_range = ranges.get(primary_sensor, 0.0)
            if primary_range <= 0.0:
                scale_factors = {s: 1.0 for s in selected_list}
            else:
                scale_factors = {
                    s: (primary_range / r) if r > 0.0 else 1.0 for s, r in ranges.items()
                }

            visible_df["scale_factor"] = visible_df["sensor"].map(scale_factors)
            visible_df["scaled_value"] = (
                visible_df["value"].astype(float) * visible_df["scale_factor"]
            )

            ts_fig = px.line(
                visible_df,
                x="timestamp",
                y="scaled_value",
                color="sensor",
                color_discrete_map=sensor_colors,
                title=(
                    f"Sensors {', '.join(selected_list)} (normalized), "
                    f"primary={primary_sensor}, method={method}"
                ),
                height=400,
            )

            # Highlight anomalies for each selected sensor (in normalized space)
            for sensor in selected_list:
                try:
                    sensor_anom = detect_anomalies(
                        data,
                        sensor_name=sensor,
                        method=method,
                        threshold=threshold,
                    )
                except ValueError:
                    # If detection fails for this sensor (e.g., insufficient data), skip it
                    continue

                mask = (sensor_anom["sensor"] == sensor) & sensor_anom["is_anomaly"]
                if not mask.any():
                    continue

                # Use indices to look up scaled values
                scaled_anom = visible_df.loc[mask & visible_df["sensor"].eq(sensor)]
                if scaled_anom.empty:
                    continue

                ts_fig.add_scatter(
                    x=scaled_anom["timestamp"],
                    y=scaled_anom["scaled_value"],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=sensor_colors.get(sensor, "#000000"),
                        line=dict(width=1, color="#000000"),
                    ),
                    name=f"{sensor} anomalies",
                )

            # Keep a stable, padded y-range based on normalized values
            y_values = (
                visible_df["scaled_value"]
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if not y_values.empty:
                y_min = float(y_values.min())
                y_max = float(y_values.max())
                pad = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
                ts_fig.update_yaxes(range=[y_min - pad, y_max + pad])

            # Annotate legend entries with scale factors so users see how each
            # sensor was normalized (multiplicative factor applied to value).
            for trace in ts_fig.data:
                sensor_name = trace.name
                if sensor_name in scale_factors:
                    factor = scale_factors[sensor_name]
                    trace.name = f"{sensor_name} (×{factor:.2f})"

            ts_fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Normalized value",
                legend_title=None,
                margin=dict(l=40, r=20, t=60, b=40),
            )
            # Enable interactive zoom/scroll on the x-axis while always
            # starting from the full time range on each update.
            ts_fig.update_xaxes(
                rangeslider=dict(visible=True),
                rangemode="tozero",
                autorange=True,
            )

        # Summary metrics figure: compute anomaly flags for all selected sensors
        annotated_frames = []
        for sensor in selected_list:
            try:
                sensor_anom = detect_anomalies(
                    data,
                    sensor_name=sensor,
                    method=method,
                    threshold=threshold,
                )
            except ValueError:
                # If detection fails (e.g. insufficient data), fall back to raw data
                sensor_anom = data.copy()
            annotated_frames.append(sensor_anom)

        if annotated_frames:
            combined_anomaly_data = pd.concat(annotated_frames, ignore_index=True)
        else:
            combined_anomaly_data = data

        metrics = summarize_metrics(combined_anomaly_data, group_by="sensor")
        metrics_df = pd.DataFrame.from_dict(metrics, orient="index").reset_index()
        metrics_df = metrics_df.rename(columns={"index": "sensor"})

        # Only show metrics for selected sensors
        metrics_df = metrics_df[metrics_df["sensor"].isin(selected_list)]

        # Separate plots: good quality (%) and anomaly rate per sensor
        sensors_order = metrics_df["sensor"].tolist()
        good_quality = metrics_df["good_quality_pct"].tolist()
        anomaly_rate = [float(v) * 100.0 for v in metrics_df["anomaly_rate"].tolist()]

        metrics_good_fig = go.Figure(
            data=[
                go.Bar(
                    x=sensors_order,
                    y=good_quality,
                    name="Good quality (%)",
                    marker_color="#1f77b4",
                )
            ]
        )
        metrics_good_fig.update_layout(
            margin=dict(l=40, r=20, t=30, b=40),
            yaxis_title="Good quality (%)",
        )

        metrics_anomaly_fig = go.Figure(
            data=[
                go.Bar(
                    x=sensors_order,
                    y=anomaly_rate,
                    name="Anomaly rate",
                    marker_color="#ff7f0e",
                )
            ]
        )
        metrics_anomaly_fig.update_layout(
            margin=dict(l=40, r=20, t=30, b=40),
            yaxis_title="Anomaly rate (%)",
        )

        # Relationships figure: show a stable correlation heatmap between sensors
        if len(selected_list) < 2:
            # Show an empty figure (no default plot) until at least two sensors are selected
            relationships_fig = px.scatter(height=320)
        else:
            try:
                rel_df = analyze_sensor_relationships(
                    data, max_lag=max_lag, sensors=selected_list
                )
            except Exception:
                rel_df = pd.DataFrame()

            if rel_df.empty:
                relationships_fig = px.scatter(
                    title="No relationships computed (insufficient data?)",
                    height=320,
                )
            else:
                rel_df = rel_df.dropna(subset=["best_corr"])
                if rel_df.empty:
                    relationships_fig = px.scatter(
                        title="No valid correlations found",
                        height=320,
                    )
                else:
                    # Build symmetric correlation matrix for a stable layout
                    sensors_sorted = sorted(selected_list)
                    corr_mat = pd.DataFrame(
                        np.eye(len(sensors_sorted)),
                        index=sensors_sorted,
                        columns=sensors_sorted,
                    )

                    for _, row in rel_df.iterrows():
                        sx = row["sensor_x"]
                        sy = row["sensor_y"]
                        if sx in corr_mat.index and sy in corr_mat.columns:
                            corr_mat.loc[sx, sy] = row["best_corr"]
                            corr_mat.loc[sy, sx] = row["best_corr"]

                    relationships_fig = px.imshow(
                        corr_mat,
                        x=corr_mat.columns,
                        y=corr_mat.index,
                        zmin=-1,
                        zmax=1,
                        color_continuous_scale="RdBu_r",
                        aspect="equal",
                        title="Sensor correlation matrix (max-abs corr with lag)",
                        height=320,
                    )
                    relationships_fig.update_layout(
                        xaxis_title="Sensor",
                        yaxis_title="Sensor",
                        margin=dict(l=40, r=20, t=60, b=40),
                    )

        return ts_fig, metrics_good_fig, metrics_anomaly_fig, relationships_fig

    return app


def main() -> None:
    app = _create_app()
    # Run on localhost:8050 by default
    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()


