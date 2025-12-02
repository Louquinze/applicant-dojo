# Implementation Notes

**Candidate Name:** Lukas Strack
**Date:** 02.12.2025  
**Time Spent:** 1.5 h

---

## 📝 Summary

Implemented a small, robust data-processing pipeline around three core functions: `ingest_data`, `detect_anomalies`, and `summarize_metrics`.  
Focused on predictable schemas, conservative validation (explicit `ValueError`s for clearly invalid input), and NaN-safe statistics so downstream steps can build on consistent data structures.  
Anomaly detection is implemented via a z-score method on a per-sensor basis, and metrics summarization exposes both basic statistics and quality/anomaly indicators keyed by sensor (and optionally time window).  
On top of this, I added a small visualization layer: a `plot.py` script for quick static views and an interactive Plotly/Dash dashboard (`dashboard.py`) that lets you explore sensors, anomaly methods, summary metrics, and cross-sensor relationships in the browser.

---

## ✅ Completed

List what you successfully implemented:

- [x] `ingest_data()` - basic functionality
- [x] `ingest_data()` - deduplication
- [x] `ingest_data()` - sorting
- [x] `ingest_data()` - validation
- [x] `detect_anomalies()` - zscore method
- [x] `detect_anomalies()` - additional methods (iqr/rolling)
- [x] `summarize_metrics()` - basic statistics
- [x] `summarize_metrics()` - quality metrics
- [x] `summarize_metrics()` - time windowing
- [x] `plot.py` - simple seaborn-based time-series plot + metrics view
- [x] `dashboard.py` - interactive Plotly/Dash app (time series, summaries, correlations)
- [x] Additional tests beyond exposed tests

---

## 🤔 Assumptions & Design Decisions

Document key assumptions and why you made certain design choices.

### Data Ingestion
- **Assumption 1:** Batches are expected to be pandas DataFrames with a fixed schema containing `timestamp`, `sensor`, `value`, `unit`, and `quality`.
  - **Rationale:** A strict schema makes downstream logic simpler and safer; failing fast on schema issues surfaces simulator/misuse bugs early.
  - **Alternative considered:** Silently dropping unexpected columns or partially-typed batches, but this can hide integration errors.

- **Assumption 2:** Duplicates can be treated as exact-row duplicates and removed via `DataFrame.drop_duplicates()`.
  - **Rationale:** This is simple, robust, and matches the exposed tests that only require “fewer rows” after deduplication.
  - **Alternative considered:** Domain-specific keys (e.g., dedupe by `timestamp` + `sensor` only), but that requires stronger domain guidance.

- **Assumption 3:** `None` or empty batches are expected and should be ignored rather than causing failures.
  - **Rationale:** The simulator can produce failed batches; ignoring empties/`None` batches makes ingestion resilient to partial outages.

- **Assumption 4:** NaNs in `value` should be preserved at ingestion time.
  - **Rationale:** Later stages (anomaly detection, imputation, or business rules) may need to differentiate “missing” from “removed”.

### Anomaly Detection
- **Method choice:** Implemented a per-sensor z-score method as the primary anomaly detector.
  - **Rationale:** Z-score is simple, well-understood, works reasonably for roughly unimodal distributions, and matches the exposed tests.

- **Threshold handling:** Use absolute z-score and flag anomalies where `abs(zscore) > threshold`.
  - **Rationale:** Keeps semantics intuitive: higher threshold → fewer anomalies; strict “greater than” avoids classifying borderline points as anomalous.
  - **Edge handling:** If the standard deviation is effectively zero, all scores are set to 0.0 so no points are treated as anomalous.

- **Missing data:** Only non-null `value`s for the target sensor contribute to mean/std; NaN z-scores remain NaN and are never flagged as anomalies.
  - **Rationale:** Prevents NaNs from skewing statistics or being mis-labelled, while still preserving those rows in the output.

### Metrics Summarization
- **Metric selection:** `mean`, `std`, `min`, `max`, `count`, `null_count`, `good_quality_pct`, and `anomaly_rate` (if anomaly flags present).
  - **Rationale:** These cover central tendency, spread, data availability, data quality, and anomaly prevalence—all core monitoring needs in an industrial setting.

- **Aggregation strategy:** Group primarily by `group_by` (default `sensor`), with optional time-window aggregation using a pandas `Grouper` on `timestamp`.
  - **Rationale:** This balances simplicity (flat dict keyed by sensor) with the ability to inspect trends over time (`sensor@timestamp` composite keys) without introducing a complex nested schema.

### Visualization / Dashboard
- **Two layers:** Kept a lightweight `plot.py` script for quick, scriptable plotting, and a richer `dashboard.py` for interactive exploration.
  - **Rationale:** The script is handy for quick checks and CI-friendly usage; the dashboard is better for exploratory analysis and demos.

- **Dashboard structure:** Time-series view at the top, followed by side-by-side summary plots (good-quality % and anomaly rate %) and a correlation heatmap.
  - **Rationale:** Mirrors a typical operator workflow: first look at raw behavior, then at health/quality, then at relationships between signals.

- **Normalization & overlay:** Time series are normalized per sensor so multiple sensors can be overlaid in a single plot without scale issues, while still preserving relative variation within each sensor.
  - **Rationale:** Makes overlaid plots readable even when sensors operate on very different numeric scales (°C vs kPa vs vibration).

- **Interactive controls:** Sensor checklist, method/threshold sliders, batch controls, and a max-lag slider for correlation.
  - **Rationale:** All controls are observable from the UI, so you can quickly test different configurations without code changes.

### Correlation / Relationships
- **Lag-aware relationships:** Implemented `analyze_sensor_relationships` to compute pairwise Pearson correlations between sensors while scanning over integer lags (e.g., \[-10, 10\] samples).
  - **Rationale:** In industrial systems, signals often respond to each other with delays; looking only at lag 0 can miss strong but shifted relationships.

- **Best-lag summary:** For each sensor pair, I store the lag that maximizes the absolute correlation (`best_lag`) and the corresponding correlation (`best_corr`).
  - **Rationale:** This gives a compact synopsis of “who moves with whom, and with what delay”, which is useful for diagnostics and causal hypotheses.

- **Dashboard view:** In the dashboard, I turn these pairwise results into a symmetric correlation matrix (heatmap) over the selected sensors, with a fixed diverging color scale \([-1, 1]\).
  - **Rationale:** The matrix is more stable and scannable than a list of bars, and the fixed color scale makes different runs visually comparable.

---

## ⚠️ Known Limitations

Be honest about what doesn't work perfectly or edge cases you didn't handle.

### Edge Cases Not Fully Handled
1. **All values for a sensor are NaN:**
   - **Impact:** Anomaly detection raises a `ValueError` due to insufficient non-null data; summarization returns NaN-based metrics (e.g., mean/std are NaN, null_count equals count).
   - **Workaround:** Upstream code can catch this and either skip anomaly detection for that sensor or impute values first.

2. **Only a single non-null value for a sensor:**
   - **Impact:** Anomaly detection also raises a `ValueError` (no meaningful variance); summarization still works but std will be 0.0.
   - **Workaround:** Treat single-point segments as “insufficient for anomaly detection” at a higher layer, or relax the requirement if business rules allow.

3. **Highly non-Gaussian / multi-modal distributions:**
   - **Impact:** Z-score may underperform or flag too many/few anomalies compared to more robust methods (e.g., IQR, robust z-score).
   - **Workaround:** Add additional methods (IQR, rolling-window, robust statistics) and choose method per sensor type.

### Performance Considerations
- **Large datasets:** Operations are vectorized pandas operations (concat, groupby, drop_duplicates), which scale reasonably but still require fitting in memory.
- **Memory usage:** `ingest_data` copies batches and concatenates them; for truly large, streaming scenarios, a chunked/iterator-based design would be preferable.

---

## 🚀 Next Steps

If you had more time, what would you improve or add?

### Priority 1: Add additional anomaly detection methods (IQR / rolling)
- **What:** Implement IQR-based and simple rolling-window methods and expose them via the existing `method` parameter.
- **Why:** Provides more robust detection for skewed or non-stationary signals and aligns with the docstring’s suggested options.
- **Estimated effort:** ~1–2 hours including tests.

### Priority 2: Make summarization output more structured for time windows
- **What:** Return a nested structure like `{sensor: {timestamp: metrics}}` instead of composite `sensor@timestamp` keys.
- **Why:** Easier to consume programmatically and more natural for time-series dashboards.
- **Estimated effort:** ~1 hour including backwards-compat helper or migration notes.

### Additional Features
- Add optional per-sensor configuration (e.g., thresholds, methods) to make anomaly detection more domain-aware.
- Introduce basic resampling/smoothing utilities so ingest + resample + detect can be run in a single, predictable pipeline.
- Extend the dashboard with simple drill-downs (e.g., click on a bar/heatmap cell to filter the time series to a specific sensor pair or time window).

### Testing & Validation
- Add property-based tests for anomaly detection to ensure reasonable behavior across a range of synthetic distributions.
- Add more integration tests around dropout-heavy simulator runs and sensors with mixed-quality flags and missing data.

---

## ❓ Questions for the Team

List any clarifying questions or areas where you'd like feedback.

1. **Requirements:** In production, should ingestion ever silently drop malformed rows, or must all such issues be surfaced as explicit errors/metrics?

2. **Design:** For duplicates, do you prefer the current conservative “exact row” dedupe or a domain-specific key (e.g., `timestamp + sensor`)?

3. **Technical:** Are there particular anomaly detection methods or libraries you use in production that you’d like this pipeline to align with?

---

## 💡 Interesting Challenges

What did you find most interesting or challenging about this exercise?

- **Most challenging:** Balancing strict validation (to avoid silent data issues) with the need to be resilient to simulator dropouts and partially bad data.
- **Most interesting:** Thinking through how anomaly detection and metrics summarization should compose, especially around NaNs and quality flags.
- **Learned:** Reinforced some practical patterns for NaN-safe statistics and how small API decisions (like output schema) impact downstream usability.

---

## 🔧 Development Environment

Document your setup for reproducibility.

- **Python version:** [e.g., 3.13.0]
- **OS:** [macOS]
- **Editor/IDE:** [Cursor]
- **Additional tools:** [Firefox browser, Plotly Dash, Seaborn]

---

## 📚 References

Any resources you consulted (documentation, articles, etc.).

---

## 💭 Final Thoughts

Any additional context you want reviewers to know.

This exercise was a good balance of data engineering and basic analytics, and I appreciated how the simulator and tests made the expectations concrete.  
I used an AI assistant to help brainstorm and refine some of the implementation patterns and documentation, but I made sure I understood and was comfortable with each design choice before keeping it.  
Given more time, I’d focus on richer anomaly methods and a more streaming-friendly ingestion pipeline, since those feel closest to how this would run in a production system.

---

**Thank you for the opportunity!** I look forward to discussing this implementation.
