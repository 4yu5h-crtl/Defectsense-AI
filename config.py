"""
DefectSense-AI: Central Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(DATA_DIR, "defectsense.db")
MODEL_PATH = os.path.join(MODELS_DIR, "defect_model.pkl")

# Simulation settings
NUM_STATIONS = 5
STATION_IDS = [f"STATION_{i+1:02d}" for i in range(NUM_STATIONS)]
NUM_RECORDS = 10000
DEFECT_RATE = 0.12  # ~12% defect rate

# Sensor baseline ranges (normal operating conditions)
SENSOR_BASELINES = {
    "temperature": {"mean": 75.0, "std": 5.0, "unit": "°C"},
    "vibration":   {"mean": 2.5,  "std": 0.8, "unit": "mm/s"},
    "pressure":    {"mean": 30.0, "std": 3.0, "unit": "PSI"},
    "cycle_time":  {"mean": 45.0, "std": 4.0, "unit": "seconds"},
}

# Feature engineering windows
ROLLING_WINDOWS = [5, 10, 20]

# ML settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_TYPE = "gradient_boosting"  # or "random_forest"

# GenAI settings
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GENAI_PROVIDER = "gemini"  # "ollama" or "gemini"
