from aws_cdk import (
    Stack,
    aws_iam as iam,
    aws_s3 as s3,
    aws_kms as kms,
    CfnOutput,
    RemovalPolicy,
    Duration
)
from aws_cdk.aws_neptunegraph import CfnGraph
from constructs import Construct


class HCPCampaignStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # Create KMS key for Neptune Analytics encryption
        neptune_kms_key = kms.Key(
            self, "HCPCampaignNeptuneKMSKey",
            description="KMS key for Neptune Analytics HCP Campaign data encryption",
            removal_policy=RemovalPolicy.DESTROY,  # For development
            enable_key_rotation=True
        )
        
        # Create S3 bucket for Neptune graph data with account suffix
        data_bucket = s3.Bucket(
            self, "HCPCampaignDataBucket",
            bucket_name=f"hcp-campaign-neptune-data-{self.account}",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,  # For development - remove for production
            versioned=False,
            public_read_access=False,
            # block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            # encryption=s3.BucketEncryption.KMS,
            # encryption_key=neptune_kms_key,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="DeleteOldVersions",
                    enabled=True,
                    expiration=Duration.days(90)  # Clean up after 90 days
                )
            ]
        )
        
        # Create Neptune Analytics Graph for HCP Campaign Data
        graph = CfnGraph(
            self, "HCPCampaignNeptuneGraph",
            graph_name="hcp-digital-campaign-graph",
            provisioned_memory=128,  # Minimum provisioned memory in GiB
            public_connectivity=True,  # Set to True if you need public access
            replica_count=0,  # Number of replicas for high availability
            deletion_protection=False,  # Set to True for production
            vector_search_configuration=CfnGraph.VectorSearchConfigurationProperty(
                vector_search_dimension=1024
            ),
            tags=[
                {
                    "key": "Environment",
                    "value": "development"
                },
                {
                    "key": "Project", 
                    "value": "HCP-Digital-Campaign"
                },
                {
                    "key": "DataType",
                    "value": "Pharmaceutical-Marketing"
                }
            ]
        )
        
        # Apply removal policy for development
        graph.apply_removal_policy(RemovalPolicy.DESTROY)
        
        # Create IAM role for Neptune Analytics access with proper trust policy
        neptune_role = iam.Role(
            self, "HCPCampaignNeptuneRole",
            assumed_by=iam.ServicePrincipal("neptune-graph.amazonaws.com"),
            inline_policies={
                "HCPCampaignNeptunePolicy": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "s3:GetObject",
                                "s3:PutObject", 
                                "s3:ListBucket"
                            ],
                            resources=[
                                data_bucket.bucket_arn,
                                f"{data_bucket.bucket_arn}/*"
                            ]
                        ),
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "kms:DescribeKey"
                            ],
                            resources=[neptune_kms_key.key_arn]
                        ),
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "kms:Decrypt",
                                "kms:GenerateDataKey"
                            ],
                            resources=[neptune_kms_key.key_arn],
                            conditions={
                                "ForAllValues:StringEquals": {
                                    "kms:EncryptionContextKeys": [
                                        "aws:neptune-graph:graphId",
                                        "aws:neptune-graph:graphExportId"
                                    ]
                                }
                            }
                        )
                    ]
                )
            }
        )
        
        # Create IAM role for data loading
        data_loader_role = iam.Role(
            self, "HCPCampaignDataLoaderRole",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("lambda.amazonaws.com"),
                iam.ServicePrincipal("glue.amazonaws.com")
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ],
            inline_policies={
                "DataLoaderPolicy": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "neptune-analytics:*"
                            ],
                            resources=[graph.attr_graph_arn]
                        ),
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "s3:GetObject",
                                "s3:PutObject",
                                "s3:ListBucket",
                                "s3:DeleteObject"
                            ],
                            resources=[
                                data_bucket.bucket_arn,
                                f"{data_bucket.bucket_arn}/*"
                            ]
                        )
                    ]
                )
            }
        )
        
        # Create SageMaker execution role for training jobs
        sagemaker_role = iam.Role(
            self, "HCPCampaignSageMakerRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess")
            ],
            inline_policies={
                "SageMakerCustomPolicy": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "neptune-analytics:GetGraph",
                                "neptune-analytics:ListGraphs",
                                "neptune-analytics:ExecuteQuery"
                            ],
                            resources=[graph.attr_graph_arn]
                        ),
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "logs:CreateLogGroup",
                                "logs:CreateLogStream",
                                "logs:PutLogEvents",
                                "logs:DescribeLogStreams"
                            ],
                            resources=["*"]
                        )
                    ]
                )
            }
        )
        
        # Grant Neptune role access to KMS key for encryption operations
        neptune_kms_key.grant(
            neptune_role,
            "kms:DescribeKey",
            "kms:Decrypt", 
            "kms:GenerateDataKey"
        )
        
        # Grant data loader role access to KMS key
        neptune_kms_key.grant_encrypt_decrypt(data_loader_role)
        
        # Grant SageMaker role access to KMS key
        neptune_kms_key.grant_encrypt_decrypt(sagemaker_role)
        
        # Grant Neptune access to S3 bucket
        data_bucket.grant_read(neptune_role)
        data_bucket.grant_read_write(data_loader_role)
        data_bucket.grant_read_write(sagemaker_role)
        
        # Output the graph details
        CfnOutput(
            self, "GraphId",
            value=graph.attr_graph_id,
            description="Neptune Analytics Graph ID for HCP Campaign Data"
        )
        
        CfnOutput(
            self, "GraphArn", 
            value=graph.attr_graph_arn,
            description="Neptune Analytics Graph ARN for HCP Campaign Data"
        )
        
        CfnOutput(
            self, "GraphName",
            value=graph.graph_name,
            description="Neptune Analytics Graph Name for HCP Campaign Data"
        )
        
        CfnOutput(
            self, "DataBucketName",
            value=data_bucket.bucket_name,
            description="S3 Bucket for Neptune graph data storage"
        )
        
        CfnOutput(
            self, "NeptuneRoleArn",
            value=neptune_role.role_arn,
            description="IAM Role ARN for Neptune Analytics access"
        )
        
        CfnOutput(
            self, "DataLoaderRoleArn",
            value=data_loader_role.role_arn,
            description="IAM Role ARN for data loading operations"
        )
        
        CfnOutput(
            self, "KMSKeyId",
            value=neptune_kms_key.key_id,
            description="KMS Key ID for Neptune Analytics encryption"
        )
        
        CfnOutput(
            self, "KMSKeyArn",
            value=neptune_kms_key.key_arn,
            description="KMS Key ARN for Neptune Analytics encryption"
        )
        
        CfnOutput(
            self, "SageMakerRoleArn",
            value=sagemaker_role.role_arn,
            description="SageMaker execution role ARN for training jobs"
        )