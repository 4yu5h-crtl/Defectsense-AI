"""Tests for the ML model training and inference modules."""
import pytest
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_simulator import generate_sensor_data
from src.model_training import train_model
from src.model_inference import predict_batch, load_model, get_high_risk_readings


@pytest.fixture(scope="module")
def training_data():
    return generate_sensor_data(num_records=2000, seed=42)


@pytest.fixture(scope="module")
def trained_artifact(training_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model.pkl")
        artifact = train_model(training_data, save_path=save_path)
        artifact["_save_path"] = save_path
        yield artifact


class TestModelTraining:
    def test_model_accuracy_above_threshold(self, trained_artifact):
        assert trained_artifact["metrics"]["accuracy"] > 0.85

    def test_model_f1_above_threshold(self, trained_artifact):
        assert trained_artifact["metrics"]["f1_score"] > 0.70

    def test_model_auc_above_threshold(self, trained_artifact):
        assert trained_artifact["metrics"]["auc_roc"] > 0.85

    def test_feature_importance_not_empty(self, trained_artifact):
        assert len(trained_artifact["feature_importance"]) > 0

    def test_feature_columns_stored(self, trained_artifact):
        assert len(trained_artifact["feature_columns"]) > 20

    def test_confusion_matrix_shape(self, trained_artifact):
        cm = trained_artifact["metrics"]["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2


class TestModelInference:
    def test_predictions_have_probability(self, training_data, trained_artifact):
        test_df = generate_sensor_data(num_records=100, seed=99)
        result = predict_batch(test_df, trained_artifact)
        assert "defect_probability" in result.columns
        assert "predicted_label" in result.columns

    def test_probabilities_in_range(self, training_data, trained_artifact):
        test_df = generate_sensor_data(num_records=100, seed=99)
        result = predict_batch(test_df, trained_artifact)
        assert (result["defect_probability"] >= 0).all()
        assert (result["defect_probability"] <= 1).all()

    def test_high_risk_filter(self, training_data, trained_artifact):
        test_df = generate_sensor_data(num_records=500, seed=99)
        result = predict_batch(test_df, trained_artifact)
        high_risk = get_high_risk_readings(result, threshold=0.7)
        assert (high_risk["defect_probability"] >= 0.7).all()

    def test_output_preserves_rows(self, training_data, trained_artifact):
        test_df = generate_sensor_data(num_records=50, seed=99)
        result = predict_batch(test_df, trained_artifact)
        assert len(result) == 50
