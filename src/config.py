from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

_raw_candidates = [
    BASE_DIR / "data" / "raw" / "Churn.csv",
    BASE_DIR / "Churn.csv",
]
RAW_DATA = next((p for p in _raw_candidates if p.exists()), _raw_candidates[0])

PROCESSED_DATA = BASE_DIR / "data" / "processed" / "clean.csv"

MODEL_PATH = BASE_DIR / "models/churn_pipeline.pkl"

TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.5
