"""
SageMaker Inference Script for Fuel Delivery Optimization
===========================================================
Handles model loading and prediction for deployed endpoints
and batch transform jobs.
"""

import json
import logging
import os
from io import StringIO
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model object (loaded once)
predictor: Optional[TimeSeriesPredictor] = None


def model_fn(model_dir: str) -> TimeSeriesPredictor:
    """Load the trained model. Called once when the endpoint starts."""
    global predictor

    logger.info(f"Loading model from {model_dir}")

    try:
        predictor = TimeSeriesPredictor.load(model_dir)
        logger.info("Model loaded successfully")
        logger.info(f"Prediction length: {predictor.prediction_length}")
        logger.info(f"Frequency: {predictor.freq}")
        return predictor
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def input_fn(request_body: str, request_content_type: str) -> TimeSeriesDataFrame:
    """
    Deserialize input data for prediction.

    Supports:
    - application/json: JSON with records or list format
    - text/csv: CSV with tank_id, reading_date, daily_consumption columns
    """
    logger.info(f"Processing input with content type: {request_content_type}")

    if request_content_type == "application/json":
        data = json.loads(request_body)

        if isinstance(data, dict):
            if "instances" in data:
                records = data["instances"]
            elif "data" in data:
                records = data["data"]
            else:
                records = data
        else:
            records = data

        if isinstance(records, list):
            df = pd.DataFrame(records)
        else:
            df = pd.DataFrame([records])

    elif request_content_type == "text/csv":
        df = pd.read_csv(StringIO(request_body))

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

    # Handle alternative column names
    column_mapping = {
        # Fuel delivery aliases
        "tank": "tank_id",
        "tank_id": "tank_id",
        "consumption": "daily_consumption",
        "daily_consumption": "daily_consumption",
        "reading_date": "reading_date",
        # Generic aliases
        "sku": "tank_id",
        "product_id": "tank_id",
        "item_id": "tank_id",
        "date": "reading_date",
        "datetime": "reading_date",
        "timestamp": "reading_date",
        "time": "reading_date",
        "demand": "daily_consumption",
        "target": "daily_consumption",
        "value": "daily_consumption",
        "sales": "daily_consumption",
    }

    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns and k != v})

    required_cols = ["tank_id", "reading_date"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["reading_date"] = pd.to_datetime(df["reading_date"])

    ts_df = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="tank_id",
        timestamp_column="reading_date"
    )

    logger.info(f"Created TimeSeriesDataFrame with {ts_df.num_items} tanks")
    return ts_df


def predict_fn(
    input_data: TimeSeriesDataFrame,
    model: TimeSeriesPredictor,
) -> pd.DataFrame:
    """Generate predictions."""
    logger.info(f"Generating predictions for {input_data.num_items} tanks")

    try:
        predictions = model.predict(input_data)
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


def output_fn(predictions: pd.DataFrame, accept: str) -> str:
    """Serialize predictions to response format."""
    logger.info(f"Serializing output with accept type: {accept}")

    if isinstance(predictions.index, pd.MultiIndex):
        output_df = predictions.reset_index()
    else:
        output_df = predictions.reset_index(drop=True)

    for col in output_df.columns:
        if pd.api.types.is_datetime64_any_dtype(output_df[col]):
            output_df[col] = output_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    if accept == "application/json" or accept == "*/*":
        result = {
            "predictions": output_df.to_dict(orient="records"),
            "metadata": {
                "num_tanks": output_df["item_id"].nunique() if "item_id" in output_df.columns else 1,
                "prediction_length": len(output_df),
                "columns": list(output_df.columns),
            },
        }
        return json.dumps(result, default=str)

    elif accept == "text/csv":
        return output_df.to_csv(index=False)

    else:
        raise ValueError(f"Unsupported accept type: {accept}")


class FuelDeliveryForecastHandler:
    """Custom handler for advanced inference scenarios."""

    def __init__(self):
        self.predictor: Optional[TimeSeriesPredictor] = None
        self.initialized = False

    def initialize(self, context: Any):
        """Initialize model (called once at startup)."""
        properties = context.system_properties
        model_dir = properties.get("model_dir", "/opt/ml/model")

        self.predictor = TimeSeriesPredictor.load(model_dir)
        self.initialized = True
        logger.info("Handler initialized successfully")

    def preprocess(self, data: List[Dict]) -> TimeSeriesDataFrame:
        """Preprocess input data."""
        if isinstance(data, str):
            data = json.loads(data)

        if isinstance(data, dict):
            if "body" in data:
                body = data["body"]
                if isinstance(body, str):
                    body = json.loads(body)
                data = body.get("instances", body.get("data", body))

        df = pd.DataFrame(data)
        df["reading_date"] = pd.to_datetime(df.get("reading_date", df.get("timestamp")))

        id_col = "tank_id" if "tank_id" in df.columns else "item_id"
        return TimeSeriesDataFrame.from_data_frame(
            df, id_column=id_col, timestamp_column="reading_date"
        )

    def inference(self, input_data: TimeSeriesDataFrame) -> pd.DataFrame:
        """Run inference."""
        return self.predictor.predict(input_data)

    def postprocess(self, predictions: pd.DataFrame) -> Dict:
        """Format predictions for response."""
        output_df = predictions.reset_index()

        for col in output_df.columns:
            if pd.api.types.is_datetime64_any_dtype(output_df[col]):
                output_df[col] = output_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "predictions": output_df.to_dict(orient="records"),
            "status": "success",
        }

    def handle(self, data: Any, context: Any) -> List[Dict]:
        """Main handler entry point."""
        if not self.initialized:
            self.initialize(context)

        try:
            input_data = self.preprocess(data)
            predictions = self.inference(input_data)
            output = self.postprocess(predictions)
            return [output]
        except Exception as e:
            logger.error(f"Error in handler: {e}")
            return [{"error": str(e), "status": "error"}]


_handler = FuelDeliveryForecastHandler()


def handle(data: Any, context: Any) -> Any:
    """Entry point for SageMaker inference container."""
    return _handler.handle(data, context)
