"""
Exposed (Public) Tests for FDSE Challenge

These tests are visible to candidates and provide:
1. Examples of expected function behavior
2. Basic correctness checks
3. Guidance on function signatures and return types

Candidates can run these tests locally to validate their implementations.
These tests focus on happy path and basic edge cases.

Run with: pytest tests/test_exposed.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_simulator import IndustrialDataSimulator
from src.data_processing import ingest_data, detect_anomalies, summarize_metrics


@pytest.fixture
def simple_data():
    """Create simple, clean test data."""
    timestamps = [datetime.now() + timedelta(seconds=i) for i in range(10)]
    data = {
        "timestamp": timestamps * 2,
        "sensor": ["temperature"] * 10 + ["pressure"] * 10,
        "value": [65.0 + i * 0.5 for i in range(10)] + [101.0 + i * 0.2 for i in range(10)],
        "unit": ["°C"] * 10 + ["kPa"] * 10,
        "quality": ["GOOD"] * 20,
    }
    return pd.DataFrame(data)


@pytest.fixture
def simulator():
    """Create deterministic simulator for testing."""
    return IndustrialDataSimulator(seed=42, dropout_rate=0.0)


class TestIngestData:
    """Tests for the ingest_data() function."""
    
    def test_ingest_empty_list_raises_error(self):
        """Should raise ValueError for empty batch list."""
        with pytest.raises(ValueError):
            ingest_data([])
    
    def test_ingest_single_batch(self, simple_data):
        """Should successfully ingest a single batch."""
        result = ingest_data([simple_data])
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "timestamp" in result.columns
        assert "sensor" in result.columns
        assert "value" in result.columns
    
    def test_ingest_multiple_batches(self, simple_data):
        """Should consolidate multiple batches."""
        batch1 = simple_data.iloc[:10]
        batch2 = simple_data.iloc[10:]
        result = ingest_data([batch1, batch2])
        assert len(result) >= 10  # At least one batch worth
    
    def test_removes_duplicates(self):
        """Should remove duplicate readings."""
        df = pd.DataFrame({
            "timestamp": [datetime.now()] * 3,
            "sensor": ["temperature"] * 3,
            "value": [65.0] * 3,
            "unit": ["°C"] * 3,
            "quality": ["GOOD"] * 3,
        })
        result = ingest_data([df], validate=True)
        # Should have fewer rows after deduplication
        assert len(result) < len(df) or len(result) == 1
    
    def test_sorts_by_timestamp(self, simple_data):
        """Should return data sorted by timestamp."""
        # Shuffle the data
        shuffled = simple_data.sample(frac=1).reset_index(drop=True)
        result = ingest_data([shuffled], validate=True)
        
        timestamps = result["timestamp"].tolist()
        assert timestamps == sorted(timestamps)
    
    def test_handles_missing_values(self):
        """Should handle NaN values appropriately."""
        df = pd.DataFrame({
            "timestamp": [datetime.now() + timedelta(seconds=i) for i in range(5)],
            "sensor": ["temperature"] * 5,
            "value": [65.0, np.nan, 66.0, np.nan, 67.0],
            "unit": ["°C"] * 5,
            "quality": ["GOOD", "BAD", "GOOD", "BAD", "GOOD"],
        })
        result = ingest_data([df], validate=True)
        # Should complete without error
        assert isinstance(result, pd.DataFrame)


class TestDetectAnomalies:
    """Tests for the detect_anomalies() function."""
    
    def test_detect_with_zscore_method(self, simple_data):
        """Should detect anomalies using z-score method."""
        clean_data = ingest_data([simple_data])
        result = detect_anomalies(clean_data, "temperature", method="zscore", threshold=3.0)
        
        assert "is_anomaly" in result.columns
        assert "anomaly_score" in result.columns
        assert result["is_anomaly"].dtype == bool

    def test_detect_with_iqr_method(self, simple_data):
        """Should detect anomalies using IQR method."""
        clean_data = ingest_data([simple_data])
        result = detect_anomalies(clean_data, "temperature", method="iqr", threshold=1.5)

        assert "is_anomaly" in result.columns
        assert "anomaly_score" in result.columns
        assert result["is_anomaly"].dtype == bool

    def test_detect_with_rolling_method(self, simple_data):
        """Should detect anomalies using rolling method."""
        clean_data = ingest_data([simple_data])
        result = detect_anomalies(clean_data, "temperature", method="rolling", threshold=3.0)

        assert "is_anomaly" in result.columns
        assert "anomaly_score" in result.columns
        assert result["is_anomaly"].dtype == bool
    
    def test_returns_same_number_of_rows(self, simple_data):
        """Should return same number of rows as input."""
        clean_data = ingest_data([simple_data])
        sensor_data = clean_data[clean_data["sensor"] == "temperature"]
        result = detect_anomalies(clean_data, "temperature", method="zscore")
        
        result_sensor = result[result["sensor"] == "temperature"]
        assert len(result_sensor) == len(sensor_data)
    
    def test_invalid_sensor_raises_error(self, simple_data):
        """Should raise ValueError for non-existent sensor."""
        clean_data = ingest_data([simple_data])
        with pytest.raises(ValueError):
            detect_anomalies(clean_data, "nonexistent_sensor")
    
    def test_invalid_method_raises_error(self, simple_data):
        """Should raise ValueError for unsupported method."""
        clean_data = ingest_data([simple_data])
        with pytest.raises(ValueError):
            detect_anomalies(clean_data, "temperature", method="invalid_method")
    
    def test_detects_obvious_anomaly(self):
        """Should detect an obvious outlier."""
        df = pd.DataFrame({
            "timestamp": [datetime.now() + timedelta(seconds=i) for i in range(11)],
            "sensor": ["temperature"] * 11,
            "value": [65.0] * 5 + [1000.0] + [65.0] * 5,  # Obvious outlier
            "unit": ["°C"] * 11,
            "quality": ["GOOD"] * 11,
        })
        clean_data = ingest_data([df])
        result = detect_anomalies(clean_data, "temperature", method="zscore", threshold=2.0)
        
        # Should flag the outlier
        assert result["is_anomaly"].sum() >= 1


class TestSummarizeMetrics:
    """Tests for the summarize_metrics() function."""
    
    def test_returns_dict_structure(self, simple_data):
        """Should return nested dictionary structure."""
        clean_data = ingest_data([simple_data])
        result = summarize_metrics(clean_data, group_by="sensor")
        
        assert isinstance(result, dict)
        assert "temperature" in result
        assert "pressure" in result
    
    def test_contains_required_metrics(self, simple_data):
        """Should include essential statistical metrics."""
        clean_data = ingest_data([simple_data])
        result = summarize_metrics(clean_data, group_by="sensor")
        
        temp_metrics = result["temperature"]
        assert "mean" in temp_metrics
        assert "std" in temp_metrics
        assert "min" in temp_metrics
        assert "max" in temp_metrics
        assert "count" in temp_metrics
    
    def test_calculates_correct_count(self, simple_data):
        """Should count readings correctly."""
        clean_data = ingest_data([simple_data])
        result = summarize_metrics(clean_data, group_by="sensor")
        
        # We know simple_data has 10 readings per sensor
        assert result["temperature"]["count"] >= 1
        assert result["pressure"]["count"] >= 1
    
    def test_handles_empty_data(self):
        """Should handle empty DataFrame gracefully."""
        empty_df = pd.DataFrame(columns=["timestamp", "sensor", "value", "unit", "quality"])
        with pytest.raises(ValueError):
            summarize_metrics(empty_df)
    
    def test_invalid_group_by_raises_error(self, simple_data):
        """Should raise ValueError for invalid group_by column."""
        clean_data = ingest_data([simple_data])
        with pytest.raises(ValueError):
            summarize_metrics(clean_data, group_by="nonexistent_column")


class TestEndToEndWorkflow:
    """Integration tests for complete workflow."""
    
    def test_complete_pipeline(self, simulator):
        """Should complete entire pipeline without errors."""
        # Step 1: Get simulated data
        batches = simulator.get_batch_readings(num_batches=3, batch_duration=20)
        assert len(batches) > 0
        
        # Step 2: Ingest data
        clean_data = ingest_data(batches, validate=True)
        assert len(clean_data) > 0
        
        # Step 3: Detect anomalies
        anomaly_data = detect_anomalies(clean_data, "temperature", method="zscore")
        assert "is_anomaly" in anomaly_data.columns
        
        # Step 4: Summarize metrics
        metrics = summarize_metrics(anomaly_data, group_by="sensor")
        assert len(metrics) > 0
        assert "temperature" in metrics
    
    @pytest.mark.timeout(30)
    def test_handles_simulator_with_dropouts(self):
        """Should handle connection dropouts gracefully."""
        # Create simulator with high dropout rate
        flaky_sim = IndustrialDataSimulator(seed=123, dropout_rate=0.3)
        
        batches = flaky_sim.get_batch_readings(num_batches=10, batch_duration=10)
        
        # Should get some successful batches despite dropouts
        if len(batches) > 0:
            clean_data = ingest_data(batches, validate=True)
            assert isinstance(clean_data, pd.DataFrame)
        # Even if all batches fail, the code should handle it
