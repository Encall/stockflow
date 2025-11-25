import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from .config import load_settings
from .model_loader import ProductionModelLoader

import yfinance as yf

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("model-serving")

settings = load_settings()
loader = ProductionModelLoader(settings)

def _get_s3_client():
    """Initialize S3/MinIO client from environment variables."""
    endpoint_url = os.getenv("AWS_S3_ENDPOINT_URL")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION", "us-east-1")
    
    if not endpoint_url or not access_key or not secret_key:
        raise ValueError(
            "MinIO credentials missing. Set AWS_S3_ENDPOINT_URL, "
            "AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY."
        )
    
    # Parse endpoint to determine if it uses HTTPS
    parsed = urlparse(endpoint_url)
    use_ssl = parsed.scheme == "https" if parsed.scheme else True
    endpoint = parsed.netloc if parsed.netloc else endpoint_url
    
    return boto3.client(
        "s3",
        endpoint_url=f"{'https' if use_ssl else 'http'}://{endpoint}",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )


def _fetch_gold_data(stock: str, n_days: int) -> pd.DataFrame:
    """Fetch the last N days of gold tier data for the specified stock from MinIO."""
    s3_client = _get_s3_client()

    bucket_name = os.getenv("AWS_S3_BUCKET", "stockflow")
    # logger.info('credentials : %s , %s , %s ', os.getenv("AWS_S3_ENDPOINT_URL"), os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
    object_key = f"gold/parquet/{stock}_data.parquet"
    
    try:
        # Download parquet file from MinIO
        logger.info("Fetching gold data for stock %s from bucket %s with key %s", stock, bucket_name, object_key)
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        parquet_data = response["Body"].read()
        
        # Read parquet data into DataFrame
        df = pd.read_parquet(BytesIO(parquet_data))
        logger.info("Successfully fetched gold data for stock %s", stock)
        
        # Sort by date (assuming there's a date column) and get last n_days
        if "date" in df.columns:
            df = df.sort_values("date", ascending=True)
        
        # Return the last N rows
        return df.tail(n_days).reset_index(drop=True)
        
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(
            status_code=404,
            detail=f"Gold tier data not found for stock '{stock}' at {object_key}",
        )
    except Exception as exc:
        logger.exception("Error fetching gold data from MinIO")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch gold data: {exc}",
        ) from exc

def _fetch_latest_data(ticker: str, end_date: str = None) -> pd.DataFrame:
        """
        Fetches stock data from yfinance up to the specified end date.
        
        Args:
            ticker: Stock ticker symbol
            end_date: End date for data fetch (YYYY-MM-DD format). 
                     If None, fetches up to today.
        
        Returns:
            DataFrame with stock data
        """
        from datetime import datetime, timedelta
        
        try:
            stock = yf.Ticker(ticker)
            
            # If end_date is None, use today
            if end_date is None:
                end_date_dt = datetime.now()
            else:
                # Parse the end_date string
                end_date_dt = pd.to_datetime(end_date)
            
            # Calculate start date (90 days before end_date to ensure we have enough data)
            start_date_dt = end_date_dt - timedelta(days=90)
            
            # Fetch data with explicit start and end dates
            df = stock.history(start=start_date_dt, end=end_date_dt)
            
            if df.empty:
                raise ValueError(f"No data returned for ticker {ticker}")
            
            # Reset index and standardize
            df = df.reset_index()
            df.columns = df.columns.str.lower()
            df = df.rename(columns={'datetime': 'date'})
            
            # Add act_symbol column as second column
            df.insert(1, 'act_symbol', ticker)
            
            # Rearrange columns: date, act_symbol, open, high, low, close, volume
            required_columns = ['date', 'act_symbol', 'open', 'high', 'low', 'close', 'volume']
            df = df[required_columns]
            
            # Ensure date is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")

app = FastAPI(
    title="StockFlow Model Serving",
    description="Serves the MLflow run tagged for production",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    stock: str
    end_date: str | None = None  # Optional end date (YYYY-MM-DD), defaults to today


class PredictionResponse(BaseModel):
    prediction: float
    run_id: str
    model_uri: str


# @app.on_event("startup")
# async def _startup_event():
#     try:
#         loader.refresh()
#         info = loader.model_info()
#         logger.info("Loaded production model: %s", info.get("model_uri"))
#     except Exception as exc:  # noqa: BLE001
#         logger.error("Failed to load production model on startup: %s", exc)


@app.get("/health")
async def health(stock: str | None = None):
    """
    Health check endpoint. 
    Optionally pass ?stock=SYMBOL to check/load a specific stock model.
    Attempts to load the model if not already cached.
    """
    print(f'Setting tracking URI: {settings.tracking_uri}')
    
    info = loader.model_info(stock=stock)
    
    # If model not in cache, try to load it
    if not info.get("model_uri"):
        try:
            logger.info(f"Model not cached for stock={stock}, attempting to load...")
            loader.get_model(stock=stock)  # This will load and cache the model
            logger.info(f"Model loaded for stock={stock}")
            info = loader.model_info(stock=stock)
            logger.info(f"Successfully loaded model: {info.get('model_uri')}")
        except Exception as exc:
            logger.error(f"Failed to load model for stock={stock}: {exc}")
            return {
                "status": "error",
                "run_id": None,
                "model_uri": None,
                "stock": stock,
                "error": str(exc),
            }
    
    return {
        "status": "ok" if info.get("model_uri") else "model_not_loaded",
        "run_id": info.get("run_id"),
        "model_uri": info.get("model_uri"),
        "stock": stock,
    }


@app.get("/metadata")
async def metadata(stock: str | None = None):
    info = loader.model_info(stock=stock)
    if not info.get("model_uri"):
        raise HTTPException(status_code=503, detail="Model not loaded")
    return jsonable_encoder(info)


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest):
    # Load the model for the specified stock
    try:
        model = loader.get_model(stock=payload.stock)
        info = loader.model_info(stock=payload.stock)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Fetch data up to the specified end_date (or today if None)
    base_frame = _fetch_latest_data(ticker=payload.stock, end_date=payload.end_date)

    # Extract only the features the model was trained with
    # Model expects: ["open", "high", "low", "volume"] in that order
    required_features = ["open", "high", "low", "volume"]
    base_frame = base_frame[-30:]
    
    try:
        # Select only required features
        feature_frame = base_frame[required_features]

        logger.info('Feature frame for prediction:\n%s', feature_frame.tail())
        
        # Model expects tensor input of shape (batch_size, seq_len, n_features)
        # Convert DataFrame to numpy array and reshape to (1, n_days, 4)
        # Must use float32 to match model signature
        feature_values = feature_frame.values.astype(np.float32)
        
        # CRITICAL: Model was trained with StandardScaler
        # Apply standardization: (x - mean) / std
        # Note: This uses the statistics from the current data window
        # Ideally, we should use the scaler fitted during training
        feature_mean = feature_values.mean(axis=0, keepdims=True)
        feature_std = feature_values.std(axis=0, keepdims=True)
        feature_scaled = (feature_values - feature_mean) / (feature_std + 1e-8)  # Add epsilon to avoid division by zero
        logger.info("feture shape before scaling: %s", feature_values.shape)
        
        feature_array = feature_scaled.reshape(1, 30, 4)

        logger.info("Feature array shape for model input: %s", feature_array.shape)
        logger.info("Feature stats - mean: %s, std: %s", feature_mean.flatten(), feature_std.flatten())
        
        # Convert to DataFrame with unnamed columns as expected by the model signature
        # The model signature expects a tensor, so we pass the numpy array directly
        raw = model.predict(feature_array)
        logger.info("Raw model output (scaled): %s", raw)
        
        # Inverse transform the prediction
        # The model predicts the scaled 'close' price
        # We need to use the target scaler's mean and std
        # Since we don't have the fitted target scaler, we approximate using 'close' price statistics
        # from the historical data (which should be similar to the training distribution)
        
        # Get close price statistics from the fetched data for descaling
        if 'close' in base_frame.columns:
            close_prices = base_frame['close'].values.astype(np.float32)
            close_mean = close_prices.mean()
            close_std = close_prices.std()
            
            # Inverse transform: prediction_original = prediction_scaled * std + mean
            if isinstance(raw, np.ndarray):
                raw_descaled = raw * close_std + close_mean
                logger.info("Descaled prediction: %s (using close_mean=%.2f, close_std=%.2f)", 
                           raw_descaled, close_mean, close_std)
            else:
                raw_descaled = raw
        else:
            logger.warning("'close' column not found in data, cannot descale prediction")
            raw_descaled = raw    
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    # Extract single prediction value
    if isinstance(raw_descaled, np.ndarray):
        # Model returns predictions, likely shape (1, 1) for single sequence prediction
        # Flatten and get first value
        prediction_value = float(raw_descaled.flatten()[0])
    else:
        prediction_value = float(raw_descaled)
    
    logger.info("Final prediction for %s: %.2f", payload.stock, prediction_value)

    return PredictionResponse(
        prediction=prediction_value,
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
