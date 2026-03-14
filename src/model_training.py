"""
ML Model Training — GradientBoosting / RandomForest classifier for defect prediction.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.feature_engineering import engineer_features, get_feature_columns


def build_model(model_type=None):
    """Create the classifier."""
    mt = model_type or config.MODEL_TYPE
    if mt == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=config.RANDOM_STATE,
        )
    else:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )


def train_model(df, model_type=None, save_path=None):
    """
    Train a defect classification model.

    Args:
        df: DataFrame with raw sensor readings (will be feature-engineered)
        model_type: 'gradient_boosting' or 'random_forest'
        save_path: Path to save the model artifact

    Returns:
        dict with model, scaler, metrics, and feature importance
    """
    featured_df = engineer_features(df)
    feature_cols = get_feature_columns(featured_df)

    X = featured_df[feature_cols].values
    y = featured_df["is_defect"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and train
    model = build_model(model_type)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="f1")

    # Feature importance
    importances = model.feature_importances_
    feat_importance = sorted(
        zip(feature_cols, importances), key=lambda x: x[1], reverse=True
    )

    metrics = {
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
        "auc_roc": auc,
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "confusion_matrix": cm.tolist(),
    }

    # Save model artifact
    save_to = save_path or config.MODEL_PATH
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_cols,
        "metrics": metrics,
        "feature_importance": feat_importance[:20],
    }
    joblib.dump(artifact, save_to)

    # Print summary
    print(f"[OK] Model trained: {type(model).__name__}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  CV F1:     {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
    print(f"  Top features: {[f[0] for f in feat_importance[:5]]}")
    print(f"  Saved to: {save_to}")

    return artifact


if __name__ == "__main__":
    from src.data_simulator import generate_sensor_data
    df = generate_sensor_data()
    train_model(df)
