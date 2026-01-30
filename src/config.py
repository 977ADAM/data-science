from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data/raw/Churn.csv"
PROCESSED_DATA = BASE_DIR / "data/processed/clean.csv"

MODEL_PATH = BASE_DIR / "models/churn_model.pkl"

TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2

print(BASE_DIR)

