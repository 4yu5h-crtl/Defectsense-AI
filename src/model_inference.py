"""
Model Inference — Load trained model and score new sensor data batches.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.feature_engineering import engineer_features, get_feature_columns


def load_model(model_path=None):
    """Load a trained model artifact."""
    path = model_path or config.MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained model at {path}. Run model_training.py first.")
    return joblib.load(path)


def predict_batch(df, artifact=None, model_path=None):
    """
    Score a batch of sensor readings for defect probability.

    Args:
        df: DataFrame with raw sensor readings
        artifact: Pre-loaded model artifact (optional)
        model_path: Path to model file (optional)

    Returns:
        DataFrame with original data plus prediction columns
    """
    if artifact is None:
        artifact = load_model(model_path)

    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_cols = artifact["feature_columns"]

    # Feature engineering
    featured_df = engineer_features(df)

    # Align feature columns (handle missing/extra columns)
    for col in feature_cols:
        if col not in featured_df.columns:
            featured_df[col] = 0.0

    X = featured_df[feature_cols].values
    X_scaled = scaler.transform(X)

    # Predict
    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = model.predict(X_scaled)

    # Build results
    result = df.copy()
    result["defect_probability"] = probabilities
    result["predicted_label"] = predictions

    return result


def predict_and_save(df, artifact=None, db_path=None):
    """Score a batch and persist predictions to the database."""
    from src.database import save_predictions

    result = predict_batch(df, artifact)

    predictions_df = result[["station_id", "timestamp", "defect_probability", "predicted_label"]].copy()
    predictions_df["reading_id"] = range(1, len(predictions_df) + 1)  # simplified ID mapping
    predictions_df["model_version"] = "1.0"

    save_predictions(predictions_df, db_path)

    defect_count = (result["predicted_label"] == 1).sum()
    print(f"[OK] Scored {len(result)} readings: {defect_count} predicted defects "
          f"({defect_count/len(result):.1%})")

    return result


def get_high_risk_readings(result_df, threshold=0.7):
    """Filter readings with defect probability above threshold."""
    return result_df[result_df["defect_probability"] >= threshold].sort_values(
        "defect_probability", ascending=False
    )


if __name__ == "__main__":
    from src.data_simulator import generate_sensor_data
    df = generate_sensor_data(num_records=100)
    result = predict_batch(df)
    high_risk = get_high_risk_readings(result)
    print(f"\nHigh-risk readings: {len(high_risk)}")
    if len(high_risk) > 0:
        print(high_risk[["station_id", "timestamp", "defect_probability"]].head())
