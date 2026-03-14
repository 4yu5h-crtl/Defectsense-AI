"""
Manufacturing Sensor Data Simulator.
Generates realistic multi-station time-series data with injected defect patterns.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _inject_defect_pattern(row, defect_type, rng):
    """Modify sensor values to create realistic defect signatures."""
    if defect_type == "thermal":
        # High temperature + high vibration → thermal stress defect
        row["temperature"] += rng.uniform(15, 30)
        row["vibration"] += rng.uniform(1.5, 4.0)
        row["pressure"] += rng.uniform(-2, 3)
    elif defect_type == "seal":
        # Pressure drop + slight vibration increase → seal failure
        row["pressure"] -= rng.uniform(8, 15)
        row["vibration"] += rng.uniform(0.5, 1.5)
        row["temperature"] += rng.uniform(2, 8)
    elif defect_type == "wear":
        # Cycle time drift + gradual vibration increase → mechanical wear
        row["cycle_time"] += rng.uniform(10, 25)
        row["vibration"] += rng.uniform(1.0, 3.0)
        row["temperature"] += rng.uniform(3, 10)
    elif defect_type == "electrical":
        # Erratic temperature spikes + pressure fluctuation
        row["temperature"] += rng.uniform(20, 40)
        row["pressure"] += rng.choice([-1, 1]) * rng.uniform(5, 12)
        row["cycle_time"] += rng.uniform(5, 15)
    return row


def generate_sensor_data(num_records=None, num_stations=None, seed=None):
    """
    Generate simulated manufacturing sensor data.

    Returns:
        pd.DataFrame with columns: station_id, timestamp, temperature,
        vibration, pressure, cycle_time, is_defect
    """
    n = num_records or config.NUM_RECORDS
    stations = (config.STATION_IDS[:num_stations] if num_stations
                else config.STATION_IDS)
    rng = np.random.default_rng(seed or config.RANDOM_STATE)

    records = []
    base_time = datetime(2025, 1, 1, 6, 0, 0)
    defect_types = ["thermal", "seal", "wear", "electrical"]

    for i in range(n):
        station = rng.choice(stations)
        timestamp = base_time + timedelta(minutes=i * 0.5)

        # Generate normal sensor readings with station-specific offsets
        station_offset = hash(station) % 5
        row = {
            "station_id": station,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": rng.normal(
                config.SENSOR_BASELINES["temperature"]["mean"] + station_offset,
                config.SENSOR_BASELINES["temperature"]["std"]
            ),
            "vibration": abs(rng.normal(
                config.SENSOR_BASELINES["vibration"]["mean"] + station_offset * 0.1,
                config.SENSOR_BASELINES["vibration"]["std"]
            )),
            "pressure": rng.normal(
                config.SENSOR_BASELINES["pressure"]["mean"] - station_offset * 0.5,
                config.SENSOR_BASELINES["pressure"]["std"]
            ),
            "cycle_time": abs(rng.normal(
                config.SENSOR_BASELINES["cycle_time"]["mean"] + station_offset * 0.3,
                config.SENSOR_BASELINES["cycle_time"]["std"]
            )),
            "is_defect": 0,
        }

        # Add temporal patterns (slight drift over time)
        time_factor = i / n
        row["temperature"] += time_factor * 2 * np.sin(2 * np.pi * time_factor * 3)
        row["vibration"] += time_factor * 0.3 * np.cos(2 * np.pi * time_factor * 5)

        # Inject defects
        if rng.random() < config.DEFECT_RATE:
            defect_type = rng.choice(defect_types, p=[0.35, 0.25, 0.25, 0.15])
            row = _inject_defect_pattern(row, defect_type, rng)
            row["is_defect"] = 1

        records.append(row)

    df = pd.DataFrame(records)

    # Ensure no negative values for physical quantities
    for col in ["vibration", "pressure", "cycle_time"]:
        df[col] = df[col].clip(lower=0.1)

    return df


def simulate_and_save(db_path=None, **kwargs):
    """Generate data and persist to SQLite."""
    from src.database import init_db, save_sensor_readings

    init_db(db_path)
    df = generate_sensor_data(**kwargs)
    save_sensor_readings(df, db_path)
    print(f"[OK] Generated {len(df)} sensor readings "
          f"({df['is_defect'].sum()} defects, "
          f"{df['is_defect'].mean():.1%} defect rate)")
    return df


if __name__ == "__main__":
    simulate_and_save()
