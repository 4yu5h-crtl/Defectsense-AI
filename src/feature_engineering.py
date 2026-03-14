"""
Feature Engineering Pipeline.
Transforms raw sensor readings into ML-ready features using rolling statistics,
anomaly indicators, and cross-sensor correlations.
"""
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SENSOR_COLS = ["temperature", "vibration", "pressure", "cycle_time"]


def add_rolling_features(df, windows=None):
    """Add rolling mean and std features for each sensor, per station."""
    windows = windows or config.ROLLING_WINDOWS
    result = df.copy()

    for station in result["station_id"].unique():
        mask = result["station_id"] == station
        station_data = result.loc[mask, SENSOR_COLS]

        for window in windows:
            for col in SENSOR_COLS:
                series = station_data[col]
                result.loc[mask, f"{col}_rolling_mean_{window}"] = (
                    series.rolling(window=window, min_periods=1).mean()
                )
                result.loc[mask, f"{col}_rolling_std_{window}"] = (
                    series.rolling(window=window, min_periods=1).std().fillna(0)
                )

    return result


def add_zscore_features(df):
    """Add z-score anomaly indicators for each sensor."""
    result = df.copy()
    for col in SENSOR_COLS:
        mean = result[col].mean()
        std = result[col].std()
        if std > 0:
            result[f"{col}_zscore"] = (result[col] - mean) / std
        else:
            result[f"{col}_zscore"] = 0.0
    return result


def add_rate_of_change(df):
    """Add rate-of-change (first derivative) features per station."""
    result = df.copy()
    for station in result["station_id"].unique():
        mask = result["station_id"] == station
        for col in SENSOR_COLS:
            result.loc[mask, f"{col}_roc"] = (
                result.loc[mask, col].diff().fillna(0)
            )
    return result


def add_cross_sensor_features(df):
    """Add interaction features between sensors."""
    result = df.copy()
    result["temp_vibration_ratio"] = result["temperature"] / result["vibration"].clip(lower=0.01)
    result["pressure_cycle_ratio"] = result["pressure"] / result["cycle_time"].clip(lower=0.01)
    result["temp_pressure_product"] = result["temperature"] * result["pressure"]
    result["vibration_cycle_product"] = result["vibration"] * result["cycle_time"]
    return result


def add_lag_features(df, lags=None):
    """Add lag features per station."""
    lags = lags or [1, 3, 5]
    result = df.copy()
    for station in result["station_id"].unique():
        mask = result["station_id"] == station
        for col in SENSOR_COLS:
            for lag in lags:
                result.loc[mask, f"{col}_lag_{lag}"] = (
                    result.loc[mask, col].shift(lag).fillna(result[col].mean())
                )
    return result


def add_time_features(df):
    """Extract time-based features from timestamp."""
    result = df.copy()
    ts = pd.to_datetime(result["timestamp"])
    result["hour"] = ts.dt.hour
    result["day_of_week"] = ts.dt.dayofweek
    result["is_night_shift"] = ((ts.dt.hour >= 22) | (ts.dt.hour < 6)).astype(int)
    return result


def add_station_encoding(df):
    """Encode station_id as numeric feature."""
    result = df.copy()
    station_map = {s: i for i, s in enumerate(sorted(result["station_id"].unique()))}
    result["station_encoded"] = result["station_id"].map(station_map)
    return result


def engineer_features(df):
    """
    Full feature engineering pipeline.

    Args:
        df: DataFrame with raw sensor readings

    Returns:
        DataFrame with all engineered features added
    """
    df = add_rolling_features(df)
    df = add_zscore_features(df)
    df = add_rate_of_change(df)
    df = add_cross_sensor_features(df)
    df = add_lag_features(df)
    df = add_time_features(df)
    df = add_station_encoding(df)

    print(f"[OK] Engineered {len(df.columns)} features from {len(SENSOR_COLS)} raw sensors")
    return df


def get_feature_columns(df):
    """Return list of feature column names (excluding metadata and target)."""
    exclude = {"id", "station_id", "timestamp", "is_defect", "reading_id"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    from src.data_simulator import generate_sensor_data
    df = generate_sensor_data(num_records=500)
    featured = engineer_features(df)
    print(f"Shape: {featured.shape}")
    print(f"Feature columns: {len(get_feature_columns(featured))}")
