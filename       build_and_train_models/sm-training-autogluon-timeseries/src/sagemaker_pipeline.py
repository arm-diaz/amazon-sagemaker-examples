"""
SageMaker Pipeline for Demand Forecasting
==========================================
Orchestrates training, deployment, and inference using Amazon SageMaker.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import boto3
import pandas as pd
import sagemaker
from sagemaker import image_uris, get_execution_role
from sagemaker.estimator import Estimator
from sagemaker.model import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemandForecastingPipeline:
    """
    End-to-end pipeline for demand forecasting with SageMaker and AutoGluon.
    """
    
    AUTOGLUON_VERSION = "1.1.1"
    
    def __init__(
        self,
        role: Optional[str] = None,
        region: Optional[str] = None,
        bucket: Optional[str] = None,
        prefix: str = "demand-forecasting",
        source_dir: Optional[str] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            role: SageMaker execution role ARN
            region: AWS region
            bucket: S3 bucket for data and model artifacts
            prefix: S3 prefix for all artifacts
            source_dir: Path to source code directory containing train.py and inference.py
        """
        self.region = region or boto3.Session().region_name or "us-east-1"
        self.session = sagemaker.Session()
        
        # Get role (from environment, IAM, or parameter)
        if role:
            self.role = role
        else:
            try:
                self.role = get_execution_role()
            except Exception:
                self.role = os.environ.get("SAGEMAKER_ROLE")
                if not self.role:
                    raise ValueError(
                        "SageMaker role not found. Provide role ARN or set SAGEMAKER_ROLE env var"
                    )
        
        self.bucket = bucket or self.session.default_bucket()
        self.prefix = prefix
        
        # Source directory for training/inference scripts
        # Must be set explicitly when running from notebook
        self.source_dir = source_dir
        
        # Clients
        self.s3 = boto3.client("s3", region_name=self.region)
        self.sm = boto3.client("sagemaker", region_name=self.region)
        
        # State
        self.training_job_name: Optional[str] = None
        self.model_name: Optional[str] = None
        self.endpoint_name: Optional[str] = None
        self.model_data: Optional[str] = None
        
        logger.info(f"Initialized pipeline in region {self.region}")
        logger.info(f"Using bucket: {self.bucket}, prefix: {self.prefix}")
        logger.info(f"AutoGluon version: {self.AUTOGLUON_VERSION}")
        if self.source_dir:
            logger.info(f"Source directory: {self.source_dir}")
    
    def _get_container_image(self, scope: str = "training", instance_type: str = "ml.m5.xlarge") -> str:
        """
        Get the correct AutoGluon container image using SageMaker SDK.
        
        Args:
            scope: "training" or "inference"
            instance_type: Instance type (determines CPU vs GPU image)
            
        Returns:
            ECR image URI
        """
        image_uri = image_uris.retrieve(
            framework="autogluon",
            region=self.region,
            version=self.AUTOGLUON_VERSION,
            image_scope=scope,
            instance_type=instance_type
        )
        logger.info(f"Using {scope} image: {image_uri}")
        return image_uri
    
    def upload_data(
        self,
        train_data: Union[str, pd.DataFrame],
        validation_data: Optional[Union[str, pd.DataFrame]] = None
    ) -> Dict[str, str]:
        """
        Upload training and validation data to S3.
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        def upload(data: Union[str, pd.DataFrame], name: str) -> str:
            key = f"{self.prefix}/data/{timestamp}/{name}.csv"
            
            if isinstance(data, pd.DataFrame):
                csv_buffer = data.to_csv(index=False)
                self.s3.put_object(Bucket=self.bucket, Key=key, Body=csv_buffer)
            else:
                self.s3.upload_file(data, self.bucket, key)
            
            s3_uri = f"s3://{self.bucket}/{key}"
            logger.info(f"Uploaded {name} data to {s3_uri}")
            return s3_uri
        
        result = {"train": upload(train_data, "train")}
        
        if validation_data is not None:
            result["validation"] = upload(validation_data, "validation")
        
        return result
    
    def train(
        self,
        train_data: Union[str, pd.DataFrame],
        validation_data: Optional[Union[str, pd.DataFrame]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        volume_size: int = 50,
        max_runtime: int = 86400,
        wait: bool = True,
        source_dir: Optional[str] = None
    ) -> str:
        """
        Train the demand forecasting model.
        
        Args:
            train_data: Training data (path or DataFrame)
            validation_data: Optional validation data
            hyperparameters: Model hyperparameters
            instance_type: SageMaker instance type
            instance_count: Number of instances
            volume_size: EBS volume size in GB
            max_runtime: Maximum training time in seconds
            wait: Wait for training to complete
            source_dir: Path to source code (overrides self.source_dir)
            
        Returns:
            Training job name
        """
        # Determine source directory
        src_dir = source_dir or self.source_dir
        if not src_dir:
            raise ValueError(
                "source_dir must be provided either in __init__() or train(). "
                "Example: pipeline = DemandForecastingPipeline(source_dir='../src')"
            )
        
        # Default hyperparameters
        hp = {
            "prediction-length": 30,
            "freq": "D",
            "presets": "medium_quality",
            "time-limit": 3600,
            "eval-metric": "MASE",
            "quantile-levels": "0.1,0.25,0.5,0.75,0.9",
            "enable-ensemble": "true",
        }
        
        if hyperparameters:
            hp.update({k.replace("_", "-"): str(v) for k, v in hyperparameters.items()})
        
        # Upload data
        data_uris = self.upload_data(train_data, validation_data)
        
        # Get training image
        training_image = self._get_container_image(
            scope="training",
            instance_type=instance_type
        )
        
        # Create estimator
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"demand-forecast-{timestamp}"
        
        estimator = Estimator(
            image_uri=training_image,
            role=self.role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size=volume_size,
            max_run=max_runtime,
            output_path=f"s3://{self.bucket}/{self.prefix}/models",
            sagemaker_session=self.session,
            hyperparameters=hp,
            entry_point="train.py",
            source_dir=src_dir,
        )
        
        # Configure input channels
        inputs = {
            "train": sagemaker.inputs.TrainingInput(
                data_uris["train"],
                content_type="text/csv"
            )
        }
        
        if "validation" in data_uris:
            inputs["validation"] = sagemaker.inputs.TrainingInput(
                data_uris["validation"],
                content_type="text/csv"
            )
        
        logger.info(f"Starting training job: {job_name}")
        logger.info(f"Instance type: {instance_type}")
        logger.info(f"Source directory: {src_dir}")
        logger.info(f"Hyperparameters: {hp}")
        
        estimator.fit(inputs, job_name=job_name, wait=wait)
        
        self.training_job_name = job_name
        self.model_data = estimator.model_data
        
        logger.info(f"Training completed. Model artifacts: {self.model_data}")
        return job_name
    
    def create_model(
        self,
        model_data: Optional[str] = None,
        model_name: Optional[str] = None,
        instance_type: str = "ml.m5.large",
        source_dir: Optional[str] = None
    ) -> str:
        """
        Create a SageMaker model from trained artifacts.
        """
        model_data = model_data or self.model_data
        if not model_data:
            raise ValueError("No model data available. Run training first or provide model_data.")
        
        src_dir = source_dir or self.source_dir
        if not src_dir:
            raise ValueError(
                "source_dir must be provided either in __init__() or create_model(). "
                "Example: pipeline = DemandForecastingPipeline(source_dir='../src')"
            )
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = model_name or f"demand-forecast-model-{timestamp}"
        
        # Get inference image
        inference_image = self._get_container_image(
            scope="inference",
            instance_type=instance_type
        )
        
        model = Model(
            image_uri=inference_image,
            model_data=model_data,
            role=self.role,
            sagemaker_session=self.session,
            entry_point="inference.py",
            source_dir=src_dir,
            name=model_name,
        )
        
        model.create()
        self.model_name = model_name
        
        logger.info(f"Created model: {model_name}")
        return model_name
    
    def deploy(
        self,
        model_name: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        instance_type: str = "ml.m5.large",
        instance_count: int = 1,
        wait: bool = True,
        source_dir: Optional[str] = None
    ) -> "DemandForecastPredictor":
        """
        Deploy model to a real-time endpoint.
        """
        model_name = model_name or self.model_name
        if not model_name:
            model_name = self.create_model(
                instance_type=instance_type,
                source_dir=source_dir
            )
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        endpoint_name = endpoint_name or f"demand-forecast-endpoint-{timestamp}"
        
        logger.info(f"Deploying model {model_name} to endpoint {endpoint_name}")
        logger.info(f"Instance type: {instance_type}, Count: {instance_count}")
        
        # Create endpoint configuration
        endpoint_config_name = f"{endpoint_name}-config"
        
        self.sm.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InstanceType": instance_type,
                    "InitialInstanceCount": instance_count,
                }
            ]
        )
        
        # Create endpoint
        self.sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
        if wait:
            logger.info("Waiting for endpoint to be ready...")
            waiter = self.sm.get_waiter("endpoint_in_service")
            waiter.wait(EndpointName=endpoint_name)
            logger.info(f"Endpoint {endpoint_name} is ready")
        
        self.endpoint_name = endpoint_name
        
        return DemandForecastPredictor(
            endpoint_name=endpoint_name,
            sagemaker_session=self.session
        )
    
    def batch_transform(
        self,
        input_data: Union[str, pd.DataFrame],
        output_path: Optional[str] = None,
        model_name: Optional[str] = None,
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        wait: bool = True,
        source_dir: Optional[str] = None
    ) -> str:
        """
        Run batch transform job for large-scale predictions.
        """
        model_name = model_name or self.model_name
        if not model_name:
            model_name = self.create_model(
                instance_type=instance_type,
                source_dir=source_dir
            )
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Upload input data if needed
        if isinstance(input_data, pd.DataFrame):
            key = f"{self.prefix}/batch-input/{timestamp}/input.csv"
            csv_buffer = input_data.to_csv(index=False)
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=csv_buffer)
            input_uri = f"s3://{self.bucket}/{key}"
        elif not input_data.startswith("s3://"):
            key = f"{self.prefix}/batch-input/{timestamp}/input.csv"
            self.s3.upload_file(input_data, self.bucket, key)
            input_uri = f"s3://{self.bucket}/{key}"
        else:
            input_uri = input_data
        
        output_path = output_path or f"s3://{self.bucket}/{self.prefix}/batch-output/{timestamp}"
        
        job_name = f"demand-forecast-batch-{timestamp}"
        
        # Create transformer
        transformer = sagemaker.transformer.Transformer(
            model_name=model_name,
            instance_count=instance_count,
            instance_type=instance_type,
            output_path=output_path,
            sagemaker_session=self.session,
            accept="application/json",
        )
        
        logger.info(f"Starting batch transform job: {job_name}")
        
        transformer.transform(
            data=input_uri,
            content_type="text/csv",
            split_type="Line",
            job_name=job_name,
            wait=wait
        )
        
        logger.info(f"Batch transform output: {output_path}")
        return output_path
    
    def cleanup(
        self,
        delete_endpoint: bool = True,
        delete_model: bool = True
    ):
        """Clean up SageMaker resources."""
        if delete_endpoint and self.endpoint_name:
            try:
                logger.info(f"Deleting endpoint: {self.endpoint_name}")
                self.sm.delete_endpoint(EndpointName=self.endpoint_name)
                self.sm.delete_endpoint_config(
                    EndpointConfigName=f"{self.endpoint_name}-config"
                )
            except Exception as e:
                logger.warning(f"Error deleting endpoint: {e}")
        
        if delete_model and self.model_name:
            try:
                logger.info(f"Deleting model: {self.model_name}")
                self.sm.delete_model(ModelName=self.model_name)
            except Exception as e:
                logger.warning(f"Error deleting model: {e}")


class DemandForecastPredictor:
    """
    Predictor for making forecasts via SageMaker endpoint.
    """
    
    def __init__(
        self,
        endpoint_name: str,
        sagemaker_session: Optional[sagemaker.Session] = None
    ):
        self.endpoint_name = endpoint_name
        self.session = sagemaker_session or sagemaker.Session()
        self.runtime = boto3.client("sagemaker-runtime")
    
    def predict(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict]],
        content_type: str = "application/json"
    ) -> pd.DataFrame:
        """
        Generate forecasts for input data.
        """
        # Prepare payload
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
            
            payload = json.dumps({"instances": df.to_dict(orient="records")})
        elif isinstance(data, list):
            payload = json.dumps({"instances": data})
        else:
            payload = json.dumps(data)
        
        # Invoke endpoint
        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType=content_type,
            Accept="application/json",
            Body=payload
        )
        
        # Parse response
        result = json.loads(response["Body"].read().decode())
        
        if "predictions" in result:
            predictions = pd.DataFrame(result["predictions"])
        else:
            predictions = pd.DataFrame(result)
        
        if "timestamp" in predictions.columns:
            predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
        
        return predictions
    
    def predict_quantiles(
        self,
        data: Union[pd.DataFrame, Dict],
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ) -> pd.DataFrame:
        """Get predictions with specified quantile intervals."""
        predictions = self.predict(data)
        
        quantile_cols = [col for col in predictions.columns 
                         if any(str(q) in col for q in quantiles)]
        
        base_cols = ["item_id", "timestamp"] if "item_id" in predictions.columns else ["timestamp"]
        
        return predictions[base_cols + quantile_cols]