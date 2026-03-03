import os
import argparse
from strands.models import BedrockModel
from mcp.client.streamable_http import streamablehttp_client 
from strands.tools.mcp.mcp_client import MCPClient
from strands import Agent
from mcp import stdio_client, StdioServerParameters
import logging

logging.getLogger("strands").setLevel(logging.DEBUG)

def create_mcp_client(neptune_endpoint):
    """Create the MCP client with provided Neptune endpoint and region."""
    return MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="uvx", 
                args=["awslabs.amazon-neptune-mcp-server@latest"],
                env={
                    "NEPTUNE_ENDPOINT": f"neptune-graph://{neptune_endpoint}",
                    "AWS_REGION": os.getenv("AWS_DEFAULT_REGION"),
                    "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION"),
                    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
                    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
                    "AWS_SESSION_TOKEN": os.getenv("AWS_SESSION_TOKEN")
                }
            )
        )
    )

def create_hcp_campaign_agent(tools, sagemaker_endpoint):
    """Create and initialize the HCP Campaign analysis agent with tools."""
    bedrockmodel = BedrockModel(
        model_id="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        temperature=0.7
    )
    
    def get_hcp_campaign_predictions(hcp_ids: list, tactic_ids: list) -> str:
        """
        Get predictive probabilities for HCP engagement from the trained SageMaker Graph Neural Network model.
        Use this tool to predict if an HCP will engage with specific marketing tactics.
        """
        import json
        import boto3
        import logging
        
        logging.getLogger("strands").info(f"Predicting engagement for HCPs {hcp_ids} and tactics {tactic_ids}")
        
        sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        
        payload = {
            "hcp_ids": hcp_ids,
            "tactic_ids": tactic_ids
        }
        
        try:
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=sagemaker_endpoint,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            result = json.loads(response['Body'].read().decode())
            return json.dumps(result, indent=2)
        except Exception as e:
            error_msg = f"Error invoking SageMaker endpoint: {str(e)}"
            logging.getLogger("strands").error(error_msg)
            # Fail safe in prod, fail loud in dev context: We'll return the error string so the agent knows it failed
            return error_msg
            
    # Include both the MCP tools and the SageMaker predictions tool
    agent_tools = tools if isinstance(tools, list) else [tools]
    agent_tools.append(get_hcp_campaign_predictions)
    
    system_prompt = """You are an expert in Healthcare Professional (HCP) digital campaign analytics and graph databases. You are provided with tools to query the HCP campaign data stored in Neptune Analytics, AND a tool to predict campaign effectiveness using SageMaker.

The graph contains the following node types and their relationships:
- **HCP**: Healthcare professionals with properties like npi, email
- **Campaign**: Marketing campaigns with properties like cpgn, cpgn_id  
- **Tactic**: Campaign tactics with properties like tact_id
- **Content**: Marketing content with properties like cntt_id, cntt_nm, url
- **Brand**: Pharmaceutical brands with properties like brd
- **Channel**: Marketing channels with properties like chnl
- **Source**: Data sources with properties like src
- **TherapyArea**: Therapeutic areas with properties like ta
- **Indication**: Medical indications with properties like indc

Key relationships:
- HCP -[ENGAGED]-> Tactic (with engagement metrics like act_cnt, act_dt)
- Brand -[HAS_CAMPAIGN]-> Campaign
- Campaign -[USES_TACTIC]-> Tactic
- Campaign -[TARGETS_TA]-> TherapyArea
- Campaign -[TARGETS_INDICATION]-> Indication
- Tactic -[RUNS_ON]-> Channel
- Tactic -[SOURCED_FROM]-> Source
- Tactic -[PROMOTES]-> Content

ALWAYS use the available tools to:
1. First explore the graph schema to understand the actual structure
2. Query the database for specific insights about HCP engagement patterns
3. If requested to make predictions about future behavior or recommend next best actions, use the SageMaker prediction tool by passing the relevant hcp_ids and tactic_ids of interest.
4. Analyze campaign effectiveness and combine historical data from the graph with predictive probabilities.
5. Provide data-driven insights about pharmaceutical marketing

Your task is to assist users in analyzing HCP digital campaign data by leveraging these tools. When users ask about engagement patterns, campaign performance, or HCP behavior, use the tools to retrieve current, accurate data rather than making assumptions.

If you cannot find information using the available tools, respond with "I don't know" rather than providing potentially incorrect information."""

    agent = Agent(
        model=bedrockmodel,
        system_prompt=system_prompt,
        tools=agent_tools,
        agent_id="hcp_campaign_agent",
        name="HCP Campaign Analytics Agent",
        description="An agent that helps users analyze HCP digital campaign data, graph patterns, and make ML predictions."
    )
    return agent

def create_initial_messages():
    """Create initial messages for the conversation."""
    return []

def main():
    """Main function to run the HCP Campaign analytics tool."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="HCP Campaign Analytics Tool")
    parser.add_argument("--neptune-endpoint", required=True, 
                       help="Neptune Analytics graph ID (e.g., g-jdct09zc09)")
    parser.add_argument("--sagemaker-endpoint", required=False, default="hcp-campaign-model-endpoint",
                       help="SageMaker Inference endpoint name")
    parser.add_argument("--region", default="us-east-1",
                       help="AWS region (default: us-east-1)")
    
    args = parser.parse_args()
    
    # Set AWS region if provided
    if args.region:
        os.environ["AWS_DEFAULT_REGION"] = args.region
    
    # Create MCP client with provided parameters
    mcp_client = create_mcp_client(args.neptune_endpoint)
    
    # Use proper context manager - this is the Pythonic way!
    with mcp_client:
        # Get tools once inside the context
        tools = mcp_client.list_tools_sync()
        
        # Create agent with tools
        hcp_campaign_agent = create_hcp_campaign_agent(tools, args.sagemaker_endpoint)
        hcp_campaign_agent.messages = create_initial_messages()
        
        print("\n🏥 HCP Digital Campaign Analytics Tool 📊\n")
        print("Ask questions about HCP engagement patterns, campaign performance, and marketing insights.")
        print("Example queries:")
        print("  - Which HCPs have the highest engagement rates?")
        print("  - What are the most effective campaign tactics?")
        print("  - Show me engagement patterns by therapeutic area")
        print("  - Which channels drive the most HCP interactions?")
        print("\nType 'exit' to quit.\n")
        
        while True:
            query = input("\nEnter your HCP campaign question> ").strip()
            
            if query.lower() == "exit":
                print("\nGoodbye! 👋")
                break
                
            if not query:
                print("Please enter a question about HCP campaign analytics.")
                continue
                
            print("\nAnalyzing HCP campaign data...\n")
            
            try:
                # Create the user message with proper format
                user_message = {
                    "role": "user",
                    "content": [{"text": query}],
                }
                
                # Add message to conversation
                hcp_campaign_agent.messages.append(user_message)
                
                # Get response
                response = hcp_campaign_agent(query)
                print(f"Campaign Analytics Results:\n{response}\n")
                
            except Exception as e:
                print(f"Error: {str(e)}\n")
            finally:
                # Reset conversation after each query
                hcp_campaign_agent.messages = create_initial_messages()

if __name__ == "__main__":
    main()