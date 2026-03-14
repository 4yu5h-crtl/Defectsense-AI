"""
GenAI Root Cause Analyzer.
Uses Ollama (local LLM) or Gemini API to generate natural language
explanations for predicted manufacturing defects.
"""
import os
import sys
import json
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _build_analysis_prompt(sensor_data, prediction_info, feature_importance=None):
    """Build the prompt for root cause analysis."""
    prompt = f"""You are a manufacturing quality engineer AI assistant. Analyze the following sensor data from a production station that has been flagged for a potential defect.

## Sensor Readings
- **Station:** {sensor_data.get('station_id', 'Unknown')}
- **Timestamp:** {sensor_data.get('timestamp', 'Unknown')}
- **Temperature:** {sensor_data.get('temperature', 0):.2f}°C (normal range: 70-80°C)
- **Vibration:** {sensor_data.get('vibration', 0):.2f} mm/s (normal range: 1.5-3.5 mm/s)
- **Pressure:** {sensor_data.get('pressure', 0):.2f} PSI (normal range: 27-33 PSI)
- **Cycle Time:** {sensor_data.get('cycle_time', 0):.2f} seconds (normal range: 41-49 seconds)

## Prediction
- **Defect Probability:** {prediction_info.get('defect_probability', 0):.1%}
- **Predicted Label:** {'DEFECT' if prediction_info.get('predicted_label', 0) == 1 else 'NORMAL'}
"""

    if feature_importance:
        prompt += "\n## Top Contributing Features\n"
        for feat, imp in feature_importance[:8]:
            prompt += f"- {feat}: {imp:.4f}\n"

    prompt += """
## Task
Provide a concise root cause analysis with the following sections:

1. **Root Cause**: Identify the most likely root cause based on the sensor patterns (1-2 sentences)
2. **Explanation**: Explain which sensor readings are abnormal and how they correlate with the predicted defect (2-3 sentences)
3. **Recommendations**: List 2-3 specific maintenance actions or process adjustments

Keep the response focused and actionable for a manufacturing engineer.
"""
    return prompt


def _call_ollama(prompt, model=None):
    """Call Ollama API for local LLM inference."""
    model = model or config.OLLAMA_MODEL
    try:
        response = requests.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Ollama is not running. Start it with 'ollama serve' or switch to Gemini provider."
        )
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


def _call_gemini(prompt, api_key=None):
    """Call Google Gemini API."""
    import google.generativeai as genai

    key = api_key or config.GEMINI_API_KEY
    if not key:
        raise ValueError("GEMINI_API_KEY not set. Set it in environment or config.py")

    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


def _fallback_analysis(sensor_data, prediction_info):
    """Rule-based fallback when no LLM is available."""
    temp = sensor_data.get("temperature", 75)
    vib = sensor_data.get("vibration", 2.5)
    pres = sensor_data.get("pressure", 30)
    cycle = sensor_data.get("cycle_time", 45)
    prob = prediction_info.get("defect_probability", 0)

    # Detect anomaly patterns
    anomalies = []
    root_cause = "Multiple sensor deviations detected"

    if temp > 90:
        anomalies.append(f"Temperature critically high at {temp:.1f}°C (normal: 70-80°C)")
    if vib > 5.0:
        anomalies.append(f"Vibration excessive at {vib:.2f} mm/s (normal: 1.5-3.5 mm/s)")
    if pres < 20:
        anomalies.append(f"Pressure critically low at {pres:.1f} PSI (normal: 27-33 PSI)")
    if cycle > 60:
        anomalies.append(f"Cycle time elevated at {cycle:.1f}s (normal: 41-49s)")

    # Determine root cause type
    if temp > 90 and vib > 5.0:
        root_cause = "Thermal stress defect — excessive heat combined with mechanical vibration indicates bearing degradation or coolant system failure."
    elif pres < 20 and vib > 3.5:
        root_cause = "Seal integrity failure — pressure drop with elevated vibration suggests a compromised seal or gasket in the hydraulic system."
    elif cycle > 60 and vib > 3.5:
        root_cause = "Mechanical wear defect — extended cycle times with increased vibration indicate progressive tool wear or drive mechanism degradation."
    elif temp > 95:
        root_cause = "Electrical/thermal anomaly — significant temperature spike suggests electrical component overheating or insufficient cooling."

    explanation = "The following sensor anomalies were detected: " + "; ".join(anomalies) if anomalies else "Subtle pattern deviations across multiple sensors suggest an emerging defect condition."

    recommendations = []
    if temp > 90:
        recommendations.append("Inspect cooling system and thermal management components")
    if vib > 5.0:
        recommendations.append("Check bearing condition and alignment of rotating components")
    if pres < 20:
        recommendations.append("Inspect seals, gaskets, and hydraulic connections for leaks")
    if cycle > 60:
        recommendations.append("Evaluate tool wear and consider scheduled replacement")
    if not recommendations:
        recommendations.append("Schedule preventive inspection of flagged station")
        recommendations.append("Review recent maintenance logs for correlated issues")

    return {
        "root_cause": root_cause,
        "explanation": explanation,
        "recommendations": "; ".join(recommendations),
        "confidence": prob,
    }


def analyze_defect(sensor_data, prediction_info, feature_importance=None, provider=None):
    """
    Generate root cause analysis for a predicted defect.

    Args:
        sensor_data: dict with sensor values
        prediction_info: dict with defect_probability and predicted_label
        feature_importance: list of (feature_name, importance) tuples
        provider: 'ollama', 'gemini', or 'fallback'

    Returns:
        dict with root_cause, explanation, recommendations, confidence
    """
    provider = provider or config.GENAI_PROVIDER
    prompt = _build_analysis_prompt(sensor_data, prediction_info, feature_importance)

    try:
        if provider == "ollama":
            response_text = _call_ollama(prompt)
        elif provider == "gemini":
            response_text = _call_gemini(prompt)
        else:
            return _fallback_analysis(sensor_data, prediction_info)
    except (ConnectionError, ValueError, RuntimeError) as e:
        print(f"[WARN] GenAI ({provider}) unavailable: {e}")
        print("  Falling back to rule-based analysis...")
        return _fallback_analysis(sensor_data, prediction_info)

    # Parse the response into structured fields
    return _parse_llm_response(response_text, prediction_info.get("defect_probability", 0))


def _parse_llm_response(text, confidence):
    """Parse LLM response into structured fields."""
    root_cause = ""
    explanation = ""
    recommendations = ""

    sections = text.split("**")
    current_section = ""

    for part in sections:
        part_lower = part.strip().lower()
        if "root cause" in part_lower:
            current_section = "root_cause"
        elif "explanation" in part_lower:
            current_section = "explanation"
        elif "recommendation" in part_lower:
            current_section = "recommendations"
        else:
            content = part.strip().strip(":").strip()
            if content:
                if current_section == "root_cause":
                    root_cause += content + " "
                elif current_section == "explanation":
                    explanation += content + " "
                elif current_section == "recommendations":
                    recommendations += content + " "

    # Fallback: if parsing didn't work well, use the full text
    if not root_cause.strip():
        lines = text.strip().split("\n")
        root_cause = lines[0] if lines else text[:200]
        explanation = text
        recommendations = "Review the analysis above and take appropriate action."

    return {
        "root_cause": root_cause.strip(),
        "explanation": explanation.strip(),
        "recommendations": recommendations.strip(),
        "confidence": confidence,
    }


def analyze_batch(predictions_df, feature_importance=None, provider=None, max_items=10):
    """Analyze multiple high-risk predictions."""
    from src.database import save_genai_report

    high_risk = predictions_df[predictions_df["predicted_label"] == 1].head(max_items)
    reports = []

    for idx, row in high_risk.iterrows():
        sensor_data = {
            "station_id": row.get("station_id", ""),
            "timestamp": row.get("timestamp", ""),
            "temperature": row.get("temperature", 0),
            "vibration": row.get("vibration", 0),
            "pressure": row.get("pressure", 0),
            "cycle_time": row.get("cycle_time", 0),
        }
        prediction_info = {
            "defect_probability": row.get("defect_probability", 0),
            "predicted_label": row.get("predicted_label", 0),
        }

        report = analyze_defect(sensor_data, prediction_info, feature_importance, provider)
        report["station_id"] = sensor_data["station_id"]
        report["timestamp"] = sensor_data["timestamp"]
        reports.append(report)

    print(f"[OK] Generated {len(reports)} root cause analyses")
    return reports


if __name__ == "__main__":
    test_sensor = {
        "station_id": "STATION_01",
        "timestamp": "2025-01-15 08:30:00",
        "temperature": 98.5,
        "vibration": 6.2,
        "pressure": 28.1,
        "cycle_time": 52.3,
    }
    test_prediction = {"defect_probability": 0.92, "predicted_label": 1}

    result = analyze_defect(test_sensor, test_prediction, provider="fallback")
    print(json.dumps(result, indent=2))
