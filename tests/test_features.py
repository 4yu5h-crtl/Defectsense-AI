"""Tests for the feature engineering module."""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_simulator import generate_sensor_data
from src.feature_engineering import (
    engineer_features, get_feature_columns,
    add_rolling_features, add_zscore_features,
    add_cross_sensor_features, add_rate_of_change,
)


@pytest.fixture
def sample_data():
    return generate_sensor_data(num_records=200, seed=42)


class TestFeatureEngineering:
    def test_output_has_more_columns(self, sample_data):
        result = engineer_features(sample_data)
        assert len(result.columns) > len(sample_data.columns)

    def test_no_nan_values(self, sample_data):
        result = engineer_features(sample_data)
        feature_cols = get_feature_columns(result)
        assert not result[feature_cols].isna().any().any()

    def test_rolling_features_created(self, sample_data):
        result = add_rolling_features(sample_data)
        assert "temperature_rolling_mean_5" in result.columns
        assert "vibration_rolling_std_10" in result.columns

    def test_zscore_features_centered(self, sample_data):
        result = add_zscore_features(sample_data)
        # Z-scores should have mean ≈ 0
        assert abs(result["temperature_zscore"].mean()) < 0.1

    def test_cross_sensor_features(self, sample_data):
        result = add_cross_sensor_features(sample_data)
        assert "temp_vibration_ratio" in result.columns
        assert "pressure_cycle_ratio" in result.columns

    def test_rate_of_change(self, sample_data):
        result = add_rate_of_change(sample_data)
        assert "temperature_roc" in result.columns

    def test_preserves_original_columns(self, sample_data):
        result = engineer_features(sample_data)
        for col in sample_data.columns:
            assert col in result.columns

    def test_feature_count_reasonable(self, sample_data):
        result = engineer_features(sample_data)
        feature_cols = get_feature_columns(result)
        assert len(feature_cols) >= 30, f"Only {len(feature_cols)} features generated"
