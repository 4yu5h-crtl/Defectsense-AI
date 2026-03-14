"""Tests for the GenAI root cause analyzer."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.genai_analyzer import analyze_defect, _build_analysis_prompt, _fallback_analysis


class TestGenAIAnalyzer:
    def test_fallback_thermal_defect(self):
        result = analyze_defect(
            {"station_id": "STATION_01", "timestamp": "2025-01-01 08:00:00",
             "temperature": 105.0, "vibration": 7.0, "pressure": 29.0, "cycle_time": 46.0},
            {"defect_probability": 0.95, "predicted_label": 1},
            provider="fallback",
        )
        assert "root_cause" in result
        assert "thermal" in result["root_cause"].lower() or "heat" in result["root_cause"].lower()
        assert result["confidence"] == 0.95

    def test_fallback_seal_failure(self):
        result = analyze_defect(
            {"station_id": "STATION_02", "timestamp": "2025-01-01 09:00:00",
             "temperature": 76.0, "vibration": 4.0, "pressure": 14.0, "cycle_time": 46.0},
            {"defect_probability": 0.80, "predicted_label": 1},
            provider="fallback",
        )
        assert "seal" in result["root_cause"].lower()

    def test_fallback_wear_defect(self):
        result = analyze_defect(
            {"station_id": "STATION_03", "timestamp": "2025-01-01 10:00:00",
             "temperature": 80.0, "vibration": 5.5, "pressure": 30.0, "cycle_time": 72.0},
            {"defect_probability": 0.75, "predicted_label": 1},
            provider="fallback",
        )
        assert "wear" in result["root_cause"].lower() or "cycle" in result["explanation"].lower()

    def test_response_structure(self):
        result = analyze_defect(
            {"station_id": "S1", "timestamp": "2025-01-01", "temperature": 100,
             "vibration": 6, "pressure": 25, "cycle_time": 55},
            {"defect_probability": 0.9, "predicted_label": 1},
            provider="fallback",
        )
        assert "root_cause" in result
        assert "explanation" in result
        assert "recommendations" in result
        assert "confidence" in result

    def test_prompt_contains_sensor_data(self):
        prompt = _build_analysis_prompt(
            {"station_id": "S1", "timestamp": "2025-01-01",
             "temperature": 99.5, "vibration": 6.2, "pressure": 18.0, "cycle_time": 65.0},
            {"defect_probability": 0.88, "predicted_label": 1},
        )
        assert "99.5" in prompt
        assert "6.2" in prompt
        assert "88.0%" in prompt

    def test_recommendations_are_actionable(self):
        result = _fallback_analysis(
            {"temperature": 105, "vibration": 7.5, "pressure": 15, "cycle_time": 70},
            {"defect_probability": 0.9},
        )
        recs = result["recommendations"]
        assert len(recs) > 10  # Not empty
        assert any(word in recs.lower() for word in ["inspect", "check", "review", "evaluate"])
