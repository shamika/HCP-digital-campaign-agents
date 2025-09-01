#!/usr/bin/env python3
"""
Cleanup SageMaker endpoints to avoid charges
"""
import boto3
import argparse

def cleanup_endpoint(endpoint_name: str, region: str):
    """Delete the endpoint to avoid charges"""
    sm_client = boto3.client('sagemaker', region_name=region)
    
    try:
        # Check if endpoint exists first
        try:
            sm_client.describe_endpoint(EndpointName=endpoint_name)
        except sm_client.exceptions.ClientError as e:
            if 'ValidationException' in str(e):
                print(f"❌ Endpoint {endpoint_name} does not exist")
                return False
            raise
        
        print(f"🗑️ Deleting endpoint: {endpoint_name}")
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"✅ Endpoint {endpoint_name} deletion initiated")
        print("💡 Note: Deletion may take a few minutes to complete")
        return True
        
    except Exception as e:
        print(f"❌ Failed to delete endpoint: {e}")
        return False

def list_endpoints(region: str, name_filter: str = None):
    """List all SageMaker endpoints"""
    sm_client = boto3.client('sagemaker', region_name=region)
    
    try:
        response = sm_client.list_endpoints()
        endpoints = response.get('Endpoints', [])
        
        if name_filter:
            endpoints = [ep for ep in endpoints if name_filter in ep['EndpointName']]
        
        if not endpoints:
            print("📋 No endpoints found")
            return []
        
        print(f"📋 Found {len(endpoints)} endpoint(s):")
        for ep in endpoints:
            status = ep['EndpointStatus']
            name = ep['EndpointName']
            created = ep['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')
            
            status_emoji = {
                'InService': '✅',
                'Creating': '🔄',
                'Updating': '🔄',
                'Deleting': '🗑️',
                'Failed': '❌',
                'OutOfService': '⚠️'
            }.get(status, '❓')
            
            print(f"  {status_emoji} {name} - {status} (created: {created})")
        
        return endpoints
        
    except Exception as e:
        print(f"❌ Failed to list endpoints: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Cleanup SageMaker endpoints")
    parser.add_argument("--region", required=True, help="AWS region")
    parser.add_argument("--endpoint-name", 
                       help="Specific endpoint name to delete")
    parser.add_argument("--list", action="store_true",
                       help="List all endpoints")
    parser.add_argument("--filter", 
                       help="Filter endpoints by name pattern (use with --list)")
    parser.add_argument("--confirm", action="store_true",
                       help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    if args.list:
        print(f"🔍 Listing endpoints in region: {args.region}")
        list_endpoints(args.region, args.filter)
        return
    
    if not args.endpoint_name:
        print("❌ Please specify --endpoint-name or use --list to see available endpoints")
        return
    
    # Show endpoint info before deletion
    print(f"🔍 Checking endpoint: {args.endpoint_name}")
    endpoints = list_endpoints(args.region, args.endpoint_name)
    
    if not endpoints:
        print(f"❌ Endpoint {args.endpoint_name} not found")
        return
    
    # Confirmation prompt
    if not args.confirm:
        response = input(f"\n⚠️  Are you sure you want to delete endpoint '{args.endpoint_name}'? (y/N): ")
        if response.lower() != 'y':
            print("❌ Deletion cancelled")
            return
    
    # Delete the endpoint
    success = cleanup_endpoint(args.endpoint_name, args.region)
    
    if success:
        print(f"\n💰 Endpoint deletion will stop billing charges")
        print("🔍 You can verify deletion with:")
        print(f"   uv run python scripts/model-training/5.cleanup_endpoint.py --list --region {args.region}")

if __name__ == "__main__":
    main()