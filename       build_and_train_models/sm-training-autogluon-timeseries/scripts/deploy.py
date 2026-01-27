#!/usr/bin/env python3
"""
Deployment Script for Demand Forecasting Model
===============================================
Deploy trained models to SageMaker endpoints.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sagemaker_pipeline import DemandForecastingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def deploy_model(args):
    """Deploy a trained model to an endpoint."""
    
    pipeline = DemandForecastingPipeline(
        role=args.role,
        region=args.region,
        bucket=args.bucket,
        prefix=args.prefix
    )
    
    # If model data URI provided, create model first
    if args.model_data:
        logger.info(f"Creating model from: {args.model_data}")
        pipeline.model_data = args.model_data
        pipeline.create_model(model_name=args.model_name)
    elif args.model_name:
        pipeline.model_name = args.model_name
    else:
        raise ValueError("Either --model-data or --model-name must be provided")
    
    # Deploy
    predictor = pipeline.deploy(
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        wait=True
    )
    
    logger.info(f"Endpoint deployed: {pipeline.endpoint_name}")
    return predictor


def delete_endpoint(args):
    """Delete an existing endpoint."""
    import boto3
    
    sm = boto3.client("sagemaker", region_name=args.region)
    
    logger.info(f"Deleting endpoint: {args.endpoint_name}")
    
    try:
        sm.delete_endpoint(EndpointName=args.endpoint_name)
        logger.info("Endpoint deleted")
    except Exception as e:
        logger.error(f"Error deleting endpoint: {e}")
    
    try:
        sm.delete_endpoint_config(EndpointConfigName=f"{args.endpoint_name}-config")
        logger.info("Endpoint config deleted")
    except Exception as e:
        logger.warning(f"Could not delete endpoint config: {e}")


def list_endpoints(args):
    """List all demand forecasting endpoints."""
    import boto3
    
    sm = boto3.client("sagemaker", region_name=args.region)
    
    response = sm.list_endpoints(
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=20,
        NameContains="demand-forecast" if not args.all else ""
    )
    
    endpoints = response.get("Endpoints", [])
    
    if not endpoints:
        logger.info("No endpoints found")
        return
    
    print("\n" + "=" * 80)
    print("SAGEMAKER ENDPOINTS")
    print("=" * 80)
    
    for ep in endpoints:
        status_color = {
            "InService": "\033[92m",  # Green
            "Creating": "\033[93m",   # Yellow
            "Failed": "\033[91m",     # Red
        }.get(ep["EndpointStatus"], "")
        
        print(f"\nName: {ep['EndpointName']}")
        print(f"Status: {status_color}{ep['EndpointStatus']}\033[0m")
        print(f"Created: {ep['CreationTime']}")
        print("-" * 40)


def update_endpoint(args):
    """Update endpoint instance configuration."""
    import boto3
    from datetime import datetime
    
    sm = boto3.client("sagemaker", region_name=args.region)
    
    # Get current endpoint config
    endpoint = sm.describe_endpoint(EndpointName=args.endpoint_name)
    current_config = endpoint["EndpointConfigName"]
    
    # Get model from current config
    config = sm.describe_endpoint_config(EndpointConfigName=current_config)
    model_name = config["ProductionVariants"][0]["ModelName"]
    
    # Create new config
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    new_config_name = f"{args.endpoint_name}-config-{timestamp}"
    
    sm.create_endpoint_config(
        EndpointConfigName=new_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": args.instance_type,
                "InitialInstanceCount": args.instance_count,
            }
        ]
    )
    
    # Update endpoint
    logger.info(f"Updating endpoint to {args.instance_type} x {args.instance_count}")
    
    sm.update_endpoint(
        EndpointName=args.endpoint_name,
        EndpointConfigName=new_config_name
    )
    
    logger.info("Endpoint update initiated")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy and manage demand forecasting endpoints"
    )
    
    # Global arguments
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--role", type=str, help="SageMaker execution role ARN")
    parser.add_argument("--bucket", type=str, help="S3 bucket")
    parser.add_argument("--prefix", type=str, default="demand-forecasting", help="S3 prefix")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy a model")
    deploy_parser.add_argument("--model-data", type=str, help="S3 URI of model artifacts")
    deploy_parser.add_argument("--model-name", type=str, help="Existing model name")
    deploy_parser.add_argument("--endpoint-name", type=str, help="Endpoint name")
    deploy_parser.add_argument("--instance-type", type=str, default="ml.m5.large")
    deploy_parser.add_argument("--instance-count", type=int, default=1)
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete an endpoint")
    delete_parser.add_argument("--endpoint-name", type=str, required=True)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List endpoints")
    list_parser.add_argument("--all", action="store_true", help="List all endpoints")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update endpoint")
    update_parser.add_argument("--endpoint-name", type=str, required=True)
    update_parser.add_argument("--instance-type", type=str, default="ml.m5.large")
    update_parser.add_argument("--instance-count", type=int, default=1)
    
    args = parser.parse_args()
    
    if args.command == "deploy":
        deploy_model(args)
    elif args.command == "delete":
        delete_endpoint(args)
    elif args.command == "list":
        list_endpoints(args)
    elif args.command == "update":
        update_endpoint(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
