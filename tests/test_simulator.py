"""Tests for the data simulator module."""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_simulator import generate_sensor_data


class TestDataSimulator:
    def test_output_shape(self):
        df = generate_sensor_data(num_records=100, seed=1)
        assert len(df) == 100
        assert set(df.columns) == {
            "station_id", "timestamp", "temperature",
            "vibration", "pressure", "cycle_time", "is_defect"
        }

    def test_station_distribution(self):
        df = generate_sensor_data(num_records=1000, seed=1)
        assert df["station_id"].nunique() == 5

    def test_defect_rate_approximate(self):
        df = generate_sensor_data(num_records=5000, seed=1)
        rate = df["is_defect"].mean()
        assert 0.05 < rate < 0.25, f"Defect rate {rate:.2%} outside expected range"

    def test_no_negative_physical_values(self):
        df = generate_sensor_data(num_records=2000, seed=1)
        assert (df["vibration"] > 0).all()
        assert (df["pressure"] > 0).all()
        assert (df["cycle_time"] > 0).all()

    def test_reproducibility(self):
        df1 = generate_sensor_data(num_records=50, seed=42)
        df2 = generate_sensor_data(num_records=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_defect_readings_have_anomalies(self):
        df = generate_sensor_data(num_records=5000, seed=1)
        defects = df[df["is_defect"] == 1]
        normals = df[df["is_defect"] == 0]
        # Defect readings should have higher avg temperature or vibration
        assert (defects["temperature"].mean() > normals["temperature"].mean() or
                defects["vibration"].mean() > normals["vibration"].mean())
