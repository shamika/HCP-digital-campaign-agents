#!/usr/bin/env python3
import aws_cdk as cdk
from hcp_campaign_stack import HCPCampaignStack


app = cdk.App()

# Create the HCP Campaign Neptune Graph stack
HCPCampaignStack(
    app, 
    "HCPCampaignStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-east-1"
    ),
    description="Neptune Analytics Graph for HCP Digital Campaign Data Analysis"
)

app.synth()