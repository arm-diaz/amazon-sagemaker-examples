# Demand Forecasting with Amazon SageMaker & AutoGluon

A production-ready demand forecasting solution using Amazon SageMaker and AutoGluon-TimeSeries for accurate, scalable predictions.

## Features

- **AutoGluon-TimeSeries**: State-of-the-art time series forecasting with automatic model selection
- **SageMaker Integration**: Managed training and inference with automatic scaling
- **Multiple Models**: Ensemble of DeepAR, ETS, ARIMA, Theta, and more
- **Probabilistic Forecasts**: Get prediction intervals, not just point forecasts
- **Easy Deployment**: Real-time endpoints or batch transform jobs

## Project Structure

```
demand-forecasting/
├── config/
│   └── config.yaml           # Configuration parameters
├── src/
│   ├── data_preparation.py   # Data loading and preprocessing
│   ├── train.py              # SageMaker training script
│   ├── inference.py          # SageMaker inference handlers
│   └── utils.py              # Utility functions
├── notebooks/
│   └── pipeline.ipynb        # End-to-end workflow notebook
├── scripts/
│   ├── deploy.py             # Deployment script
│   └── generate_sample_data.py # Generate synthetic data
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data (Optional)

```bash
python scripts/generate_sample_data.py
```

### 3. Configure

Edit `config/config.yaml` with your AWS settings and model parameters.

### 4. Run Training

```python
from src.sagemaker_pipeline import DemandForecastingPipeline

pipeline = DemandForecastingPipeline()
pipeline.train()
```

### 5. Deploy & Predict

```python
predictor = pipeline.deploy()
forecasts = predictor.predict(data)
```

## Configuration

Key parameters in `config/config.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `prediction_length` | Forecast horizon | 30 |
| `freq` | Time series frequency | D (daily) |
| `time_limit` | Training time limit (seconds) | 3600 |
| `presets` | AutoGluon quality preset | medium_quality |

## Model Architecture

The solution uses AutoGluon's ensemble approach:

1. **Statistical Models**: ETS, ARIMA, Theta, Seasonal Naive
2. **Machine Learning**: LightGBM, CatBoost
3. **Deep Learning**: DeepAR, Temporal Fusion Transformer
4. **Ensemble**: Weighted combination for optimal accuracy

## Input Data Format

Your data should have these columns:

| Column | Type | Description |
|--------|------|-------------|
| `item_id` | string | Product/SKU identifier |
| `timestamp` | datetime | Date/time of observation |
| `target` | float | Demand value |
| `static_features` | dict (optional) | Item-level features |
| `dynamic_features` | dict (optional) | Time-varying features |

## License

MIT License
