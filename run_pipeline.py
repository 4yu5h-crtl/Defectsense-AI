"""
DefectSense-AI: End-to-End Pipeline Runner
Runs: simulate → feature engineer → train → predict → analyze
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.database import init_db
from src.data_simulator import generate_sensor_data
from src.feature_engineering import engineer_features
from src.model_training import train_model
from src.model_inference import predict_batch, get_high_risk_readings
from src.genai_analyzer import analyze_batch


def run_pipeline(num_records=5000, genai_provider="fallback", max_analyses=10, seed=42):
    """Execute the full DefectSense-AI pipeline."""
    print("=" * 60)
    print("  DefectSense-AI: Manufacturing Defect Prediction Pipeline")
    print("=" * 60)
    start = time.time()

    # Step 1: Initialize database
    print("\n[1/5] Initializing database...")
    init_db()
    print("[OK] Database ready")

    # Step 2: Simulate sensor data
    print(f"\n[2/5] Simulating {num_records} sensor readings...")
    df = generate_sensor_data(num_records=num_records, seed=seed)
    print(f"  Stations: {df['station_id'].nunique()}")
    print(f"  Defect rate: {df['is_defect'].mean():.1%}")

    # Step 3: Train model
    print("\n[3/5] Training ML model...")
    artifact = train_model(df)

    # Step 4: Run predictions
    print("\n[4/5] Running predictions...")
    result = predict_batch(df, artifact)
    high_risk = get_high_risk_readings(result, threshold=0.5)
    print(f"  Predicted defects: {(result['predicted_label']==1).sum()}")
    print(f"  High-risk readings (>50%): {len(high_risk)}")

    # Step 5: GenAI analysis
    print(f"\n[5/5] Generating root cause analyses ({genai_provider})...")
    reports = analyze_batch(
        result,
        feature_importance=artifact.get("feature_importance"),
        provider=genai_provider,
        max_items=max_analyses,
    )

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  {len(df)} readings → {len(reports)} root cause analyses")
    print("=" * 60)

    # Print sample report
    if reports:
        r = reports[0]
        print(f"\n[Sample Analysis] ({r.get('station_id', 'N/A')}):")
        print(f"   Root Cause: {r.get('root_cause', 'N/A')[:120]}")
        print(f"   Actions:    {r.get('recommendations', 'N/A')[:120]}")

    return {"data": df, "artifact": artifact, "predictions": result, "reports": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DefectSense-AI Pipeline")
    parser.add_argument("--records", type=int, default=5000, help="Number of sensor records")
    parser.add_argument("--provider", choices=["ollama", "gemini", "fallback"], default="fallback")
    parser.add_argument("--analyses", type=int, default=10, help="Max GenAI analyses")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_pipeline(
        num_records=args.records,
        genai_provider=args.provider,
        max_analyses=args.analyses,
        seed=args.seed,
    )
