import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pathlib import Path
import logging

from app.schemas import PredictRequest, PredictResponse
from src.config import MODEL_PATH

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = MODEL_PATH

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.pipe = joblib.load(str(PIPELINE_PATH))
    except Exception:
        app.state.pipe = None
        logger.exception("Failed to load pipeline from %s", PIPELINE_PATH)
    yield

app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    pipe = getattr(app.state, "pipe", None)
    return {
        "status": "ok" if pipe is not None else "pipeline_not_loaded",
        "pipeline_path": str(PIPELINE_PATH),
        "pipeline_loaded": pipe is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pipe = getattr(app.state, "pipe", None)
    if pipe is None:
        logger.error("Pipeline is not loaded, cannot serve /predict")
        raise HTTPException(status_code=500, detail="Pipeline not loaded")

    # pydantic v2: model_dump; pydantic v1: dict
    customer_obj = req.customer
    payload = customer_obj.model_dump() if hasattr(customer_obj, "model_dump") else customer_obj.dict()
    # строго один клиент
    df = pd.DataFrame([payload])
    proba = float(pipe.predict_proba(df)[:, 1].item())
    return PredictResponse(churn_probability=proba)
