#!/usr/bin/env python3
"""
Deploy the trained HCP campaign model for inference
"""
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import argparse

def deploy_model(model_data_url: str, role_arn: str, region: str, endpoint_name: str = "hcp-campaign-model", 
                instance_type: str = "ml.m5.large", framework_version: str = "2.2", python_version: str = "py310"):
    """
    Deploy the trained model to a SageMaker endpoint
    
    Args:
        model_data_url: S3 URL to the model.tar.gz from training job
        role_arn: SageMaker execution role ARN
        region: AWS region
        endpoint_name: Name for the endpoint
        instance_type: Instance type for inference
        framework_version: PyTorch framework version
        python_version: Python version
    """
    
    # Set up AWS session with region
    session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(boto_session=session)
    
    # Create PyTorch model
    pytorch_model = PyTorchModel(
        model_data=model_data_url,
        role=role_arn,
        entry_point="inference.py",  # We'll create this
        source_dir="sm",
        framework_version=framework_version,
        py_version=python_version,
        sagemaker_session=sagemaker_session
    )
    
    # Deploy to endpoint
    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )
    
    print(f"✅ Model deployed to endpoint: {endpoint_name}")
    print(f"Endpoint URL: {predictor.endpoint_name}")
    
    return predictor

def get_model_url_from_training_output():
    """
    Helper to show where to find the model URL from SageMaker training output
    """
    print("📋 To find your model URL:")
    print("1. Check your SageMaker training job output")
    print("2. Look for 'Training job completed' message")
    print("3. The model.tar.gz will be in: s3://sagemaker-{region}-{account}/pytorch-training-{timestamp}/output/model.tar.gz")
    print("\nExample model URL format:")
    print("s3://sagemaker-us-east-1-709753484661/pytorch-training-2025-08-31-19-56-16-265/output/model.tar.gz")
    return None

def test_endpoint(predictor, sample_hcp_ids=None, sample_tactic_ids=None):
    """
    Test the deployed endpoint with sample data
    
    Args:
        predictor: SageMaker predictor instance
        sample_hcp_ids: List of HCP IDs to test
        sample_tactic_ids: List of tactic IDs to test
    """
    if sample_hcp_ids is None:
        sample_hcp_ids = [0, 1, 2]  # Sample HCP node IDs
    
    if sample_tactic_ids is None:
        sample_tactic_ids = [0, 1, 2]  # Sample tactic node IDs
    
    test_data = {
        "hcp_ids": sample_hcp_ids,
        "tactic_ids": sample_tactic_ids
    }
    
    try:
        # Set the serializer and deserializer to handle JSON properly
        from sagemaker.serializers import JSONSerializer
        from sagemaker.deserializers import JSONDeserializer
        
        predictor.serializer = JSONSerializer()
        predictor.deserializer = JSONDeserializer()
        
        result = predictor.predict(test_data)
        print("✅ Endpoint test successful!")
        print(f"Prediction result: {result}")
        return result
    except Exception as e:
        print(f"❌ Endpoint test failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Deploy HCP campaign model to SageMaker endpoint")
    parser.add_argument("--model-url", required=True, 
                       help="S3 URL to model.tar.gz from training job")
    parser.add_argument("--region", required=True, help="AWS region")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--endpoint-name", default="hcp-campaign-model-endpoint",
                       help="Name for the endpoint (default: hcp-campaign-model-endpoint)")
    parser.add_argument("--instance-type", default="ml.m5.large",
                       help="Instance type for inference (default: ml.m5.large)")
    parser.add_argument("--framework-version", default="2.2",
                       help="PyTorch framework version (default: 2.2)")
    parser.add_argument("--python-version", default="py310",
                       help="Python version (default: py310)")
    parser.add_argument("--test", action="store_true",
                       help="Test the endpoint after deployment")
    
    args = parser.parse_args()
    
    print("🚀 Deploying HCP Campaign Model...")
    print(f"📦 Model URL: {args.model_url}")
    print(f"🏷️ Endpoint name: {args.endpoint_name}")
    print(f"💻 Instance type: {args.instance_type}")
    
    # Deploy the model
    predictor = deploy_model(
        model_data_url=args.model_url,
        role_arn=args.role_arn,
        region=args.region,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        framework_version=args.framework_version,
        python_version=args.python_version
    )
    
    # Test the endpoint if requested
    if args.test:
        print("\n🧪 Testing endpoint...")
        test_endpoint(predictor)
    
    print(f"\n📝 Endpoint deployed: {args.endpoint_name}")
    print("💡 To delete the endpoint when done:")
    print(f"   uv run python scripts/model-training/5.cleanup_endpoint.py --endpoint-name {args.endpoint_name} --region {args.region}")

if __name__ == "__main__":
    main()