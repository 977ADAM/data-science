import pandas as pd
from .config import RAW_DATA

def load_data():
    return pd.read_csv(RAW_DATA)
