from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import sagemaker
import boto3

# Set up AWS session with region
region = "us-east-1"  # Change this to your preferred region
session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.Session(boto_session=session)

role = "arn:aws:iam::709753484661:role/HCPCampaignStack-HCPCampaignSageMakerRole35E854E3-JUxrTwp1BLx4"

# Your prepared/processed files in S3:
# s3://<YOUR_BUCKET>/graph-processed/hcp/vertices_hcp.csv
# s3://<YOUR_BUCKET>/graph-processed/hcp/vertices_tactic.csv
# s3://<YOUR_BUCKET>/graph-processed/hcp/edges_engaged.csv

est = PyTorch(
    entry_point="train_dgl_lp.py",
    source_dir="sm",                 # folder containing the script
    role=role,
    framework_version="2.2",
    py_version="py310",
    instance_count=1,
    instance_type="ml.g5.2xlarge",
    sagemaker_session=sagemaker_session,  # Use the session with region
    hyperparameters={               # no bucket/prefix here
        "epochs": 3,
        "hidden": 128,
        "neg": 4
    },
)

train_input = TrainingInput(
    s3_data="s3://hcp-campaign-neptune-data-709753484661/neptune/dgl-processed",  # folder with the 3 CSVs
    input_mode="File"                                    # SageMaker downloads to container
)

est.fit({"train": train_input})