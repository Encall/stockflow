import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from .config import load_settings
from .model_loader import ProductionModelLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("model-serving")

settings = load_settings()
loader = ProductionModelLoader(settings)

app = FastAPI(
    title="StockFlow Model Serving",
    description="Serves the MLflow run tagged for production",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    instances: List[Dict[str, Any]]
    stock: str | None = None
    horizons: List[int] | None = None  # explicit horizons, e.g., [1, 5, 10]
    horizon_days: int | None = None  # convenience: expand to range 1..horizon_days


class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    run_id: str
    model_uri: str


@app.on_event("startup")
async def _startup_event():
    try:
        loader.refresh()
        info = loader.model_info()
        logger.info("Loaded production model: %s", info.get("model_uri"))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load production model on startup: %s", exc)


@app.get("/health")
async def health():
    info = loader.model_info()
    return {
        "status": "ok" if info.get("model_uri") else "model_not_loaded",
        "run_id": info.get("run_id"),
        "model_uri": info.get("model_uri"),
    }


@app.get("/metadata")
async def metadata(stock: str | None = None):
    info = loader.model_info(stock=stock)
    if not info.get("model_uri"):
        raise HTTPException(status_code=503, detail="Model not loaded")
    return jsonable_encoder(info)


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest):
    try:
        model = loader.get_model(stock=payload.stock)
        info = loader.model_info(stock=payload.stock)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not payload.instances:
        raise HTTPException(status_code=400, detail="instances must not be empty")

    # Build horizon list: explicit horizons or 1..N if horizon_days is provided
    if payload.horizons and payload.horizon_days:
        raise HTTPException(
            status_code=400,
            detail="Provide either horizons or horizon_days, not both",
        )
    if payload.horizons:
        horizon_values = sorted(set(payload.horizons))
    elif payload.horizon_days:
        if payload.horizon_days < 1:
            raise HTTPException(
                status_code=400, detail="horizon_days must be at least 1"
            )
        horizon_values = list(range(1, payload.horizon_days + 1))
    else:
        horizon_values = [1]

    try:
        base_frame = pd.DataFrame(payload.instances)
        frames = []
        for horizon in horizon_values:
            df_h = base_frame.copy()
            df_h["horizon"] = horizon
            frames.append(df_h)
        predict_frame = pd.concat(frames, ignore_index=True)

        raw = model.predict(predict_frame)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    if isinstance(raw, np.ndarray):
        raw = raw.tolist()

    if len(raw) != len(predict_frame):
        raise HTTPException(
            status_code=500,
            detail="Prediction length mismatch with input rows",
        )

    # Group predictions per horizon (preserving original instance order within each horizon)
    grouped: Dict[int, List[Any]] = {h: [] for h in horizon_values}
    rows_per_instance = len(base_frame)
    for idx, horizon in enumerate(horizon_values):
        start = idx * rows_per_instance
        end = start + rows_per_instance
        grouped[horizon] = list(raw[start:end])

    horizons_output = [
        {"horizon": h, "predictions": grouped[h]} for h in horizon_values
    ]

    return PredictionResponse(
        predictions=horizons_output,
        run_id=info.get("run_id"),
        model_uri=info.get("model_uri"),
    )


@app.post("/reload")
async def reload_model(stock: str | None = None):
    try:
        loader.refresh(stock=stock)
        info = loader.model_info(stock=stock)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}") from exc
    return jsonable_encoder(
        {"status": "reloaded", "run_id": info.get("run_id"), "stock": stock}
    )
