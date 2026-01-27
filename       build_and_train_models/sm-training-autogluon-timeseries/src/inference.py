"""
SageMaker Inference Script for Demand Forecasting
==================================================
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
    """
    Load the trained model.
    
    This function is called once when the endpoint starts or when
    a batch transform job begins.
    
    Args:
        model_dir: Directory containing the model artifacts
        
    Returns:
        Loaded TimeSeriesPredictor
    """
    global predictor
    
    logger.info(f"Loading model from {model_dir}")
    
    try:
        predictor = TimeSeriesPredictor.load(model_dir)
        logger.info(f"Model loaded successfully")
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
    - text/csv: CSV with item_id, timestamp, target columns
    
    Args:
        request_body: Raw request body
        request_content_type: Content type header
        
    Returns:
        TimeSeriesDataFrame for prediction
    """
    logger.info(f"Processing input with content type: {request_content_type}")
    
    if request_content_type == "application/json":
        data = json.loads(request_body)
        
        # Handle different JSON formats
        if isinstance(data, dict):
            # Check for wrapped format: {"instances": [...]}
            if "instances" in data:
                records = data["instances"]
            elif "data" in data:
                records = data["data"]
            else:
                # Single record or list in dict
                records = data
        else:
            records = data
        
        # Convert to DataFrame
        if isinstance(records, list):
            df = pd.DataFrame(records)
        else:
            df = pd.DataFrame([records])
    
    elif request_content_type == "text/csv":
        df = pd.read_csv(StringIO(request_body))
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")
    
    # Ensure required columns exist
    required_cols = ["item_id", "timestamp"]
    
    # Handle alternative column names
    column_mapping = {
        "sku": "item_id",
        "product_id": "item_id",
        "date": "timestamp",
        "datetime": "timestamp",
        "time": "timestamp",
        "demand": "target",
        "value": "target",
        "sales": "target",
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Create TimeSeriesDataFrame
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="timestamp"
    )
    
    logger.info(f"Created TimeSeriesDataFrame with {ts_df.num_items} items")
    return ts_df


def predict_fn(
    input_data: TimeSeriesDataFrame,
    model: TimeSeriesPredictor
) -> pd.DataFrame:
    """
    Generate predictions.
    
    Args:
        input_data: TimeSeriesDataFrame with historical data
        model: Loaded TimeSeriesPredictor
        
    Returns:
        DataFrame with predictions
    """
    logger.info(f"Generating predictions for {input_data.num_items} items")
    
    try:
        # Generate predictions
        predictions = model.predict(input_data)
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


def output_fn(predictions: pd.DataFrame, accept: str) -> str:
    """
    Serialize predictions to response format.
    
    Args:
        predictions: Prediction DataFrame
        accept: Accept header from request
        
    Returns:
        Serialized predictions
    """
    logger.info(f"Serializing output with accept type: {accept}")
    
    # Reset index for serialization
    if isinstance(predictions.index, pd.MultiIndex):
        output_df = predictions.reset_index()
    else:
        output_df = predictions.reset_index(drop=True)
    
    # Convert timestamps to strings
    for col in output_df.columns:
        if pd.api.types.is_datetime64_any_dtype(output_df[col]):
            output_df[col] = output_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    if accept == "application/json" or accept == "*/*":
        result = {
            "predictions": output_df.to_dict(orient="records"),
            "metadata": {
                "num_items": output_df["item_id"].nunique() if "item_id" in output_df.columns else 1,
                "prediction_length": len(output_df),
                "columns": list(output_df.columns)
            }
        }
        return json.dumps(result, default=str)
    
    elif accept == "text/csv":
        return output_df.to_csv(index=False)
    
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


# Alternative handler for custom inference logic
class DemandForecastHandler:
    """
    Custom handler for advanced inference scenarios.
    """
    
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
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        return TimeSeriesDataFrame.from_data_frame(
            df,
            id_column="item_id",
            timestamp_column="timestamp"
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
            "status": "success"
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


# Create handler instance for SageMaker
_handler = DemandForecastHandler()


def handle(data: Any, context: Any) -> Any:
    """Entry point for SageMaker inference container."""
    return _handler.handle(data, context)
