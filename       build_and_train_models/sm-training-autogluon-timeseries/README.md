# Fuel Delivery Optimization with Amazon SageMaker & AutoGluon

Replace the simple heuristic (`avg_consumption * 1.20`) with ML-driven delivery
recommendations using **AutoGluon-TimeSeries** for consumption forecasting and
**Amazon SageMaker** for scalable training and inference.

## Overview

This project forecasts daily fuel consumption per tank and converts those
forecasts into optimised delivery schedules. It compares the ML approach against
the heuristic baseline across business metrics such as fill rate, emergency
delivery rate, and volume utilisation.

### Key Features

- **Consumption Forecasting** - 10-day horizon using AutoGluon ensemble models
- **Delivery Optimization** - Calculates optimal delivery volume, timing, and urgency
- **Multi-Tank Scheduling** - Prioritises deliveries across a fleet with capacity limits
- **Heuristic vs ML Comparison** - Quantifies improvement over the `avg * 1.20` rule
- **Walk-Forward Backtesting** - Validates strategy on historical data
- **US Holiday Calendar** - Federal + configurable state holidays via `holidays` library

## Project Structure

```
.
├── config/
│   └── config.yaml            # All configuration (columns, optimization, AWS)
├── data/                      # Generated / user-provided CSV data
├── models/                    # Local model artifacts
├── notebooks/
│   └── pipeline.ipynb         # Primary deliverable - full end-to-end notebook
├── scripts/
│   ├── generate_sample_data.py
│   └── deploy.py
├── src/
│   ├── __init__.py
│   ├── backtesting.py         # Walk-forward backtest engine
│   ├── calendar_us.py         # US holiday utilities
│   ├── data_preparation.py    # FuelDeliveryDataProcessor
│   ├── inference.py           # SageMaker inference handler
│   ├── optimization.py        # DeliveryOptimizer & DeliveryScheduler
│   ├── sagemaker_pipeline.py  # FuelDeliveryPipeline (SageMaker orchestration)
│   ├── train.py               # SageMaker training script
│   └── utils.py               # Metrics, data gen, visualisation, AWS helpers
├── main.py                    # CLI entry point (demo, train, deploy, generate)
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data

```bash
python scripts/generate_sample_data.py --tanks 50 --days 365
```

### 3. Run Demo (Local Training + Optimization)

```bash
python main.py --mode demo
```

### 4. Run the Notebook

Open `notebooks/pipeline.ipynb` in SageMaker Studio or JupyterLab and execute
all cells. Sections 1-5 run locally; sections 6-7 require AWS credentials.

## Domain Mapping

| Column | Description | Example |
|--------|-------------|---------|
| `tank_id` | Unique tank identifier | `TANK_0001` |
| `reading_date` | Observation date | `2024-06-15` |
| `daily_consumption` | Target variable (gallons/day) | `5.23` |
| `tank_type` | Category: residential, commercial, industrial | `residential` |
| `tank_capacity` | Tank capacity in gallons | `275` |
| `current_level` | Current fuel level in gallons | `180.5` |

## Tank Types

| Type | Consumption (gal/day) | Capacity (gal) |
|------|----------------------|-----------------|
| Residential | 2 - 8 | 275 |
| Commercial | 8 - 25 | 1,000 |
| Industrial | 25 - 60 | 5,000 |

## Configuration

All settings are in `config/config.yaml`:

- **Optimization**: min/max delivery volumes, safety buffers, fill targets, reorder thresholds
- **Holidays**: `country: US`, optional `state` for state-specific holidays
- **Model**: `prediction_length: 10`, presets, time limits
- **AWS**: region, S3 prefix, instance types

## SageMaker Training

```bash
python main.py --mode sagemaker --train-path data/train.csv --deploy
```

Or from Python:

```python
from src.sagemaker_pipeline import FuelDeliveryPipeline

pipeline = FuelDeliveryPipeline(source_dir="src")
pipeline.train(train_data="data/train.csv")
predictor = pipeline.deploy()
```

## Model Architecture

The solution uses AutoGluon's ensemble approach:

1. **Statistical Models**: ETS, ARIMA, Theta, Seasonal Naive
2. **Machine Learning**: LightGBM, CatBoost (via RecursiveTabular / DirectTabular)
3. **Deep Learning**: DeepAR, Temporal Fusion Transformer
4. **Ensemble**: Weighted combination for optimal accuracy

## Requirements

- Python 3.9+
- AutoGluon TimeSeries >= 1.1.0
- SageMaker SDK >= 2.200.0
- holidays >= 0.25
- See `requirements.txt` for full list

## License

MIT License
