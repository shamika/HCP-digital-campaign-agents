import boto3
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Export HCP campaign data from Neptune Analytics")
    parser.add_argument("--region", required=True, help="AWS region")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket name for export")
    parser.add_argument("--s3-prefix", default="neptune/hcp_marketing-exported/", 
                       help="S3 prefix for exported files")
    parser.add_argument("--role-arn", required=True, help="IAM role ARN for Neptune export")
    parser.add_argument("--graph-id", required=True, help="Neptune Analytics graph ID")
    parser.add_argument("--kms-key-arn", required=True, help="KMS key ARN for encryption")
    parser.add_argument("--format", default="CSV", choices=["CSV", "PARQUET"],
                       help="Export format (default: CSV)")
    
    args = parser.parse_args()
    
    # Configure the Neptune Analytics client
    client = boto3.client('neptune-graph', region_name=args.region)
    
    print(f"🚀 Starting Neptune Analytics export...")
    print(f"Graph ID: {args.graph_id}")
    print(f"Destination: s3://{args.s3_bucket}/{args.s3_prefix}")
    print(f"Format: {args.format}")
    
    response = client.start_export_task(
        graphIdentifier=args.graph_id,
        roleArn=args.role_arn,
        format=args.format,
        destination=f"s3://{args.s3_bucket}/{args.s3_prefix}",
        kmsKeyIdentifier=args.kms_key_arn
    )

    task_id = response['taskId']
    print(f"✅ Export started—the task ID is: {task_id}")

    while True:
        status_resp = client.list_export_tasks(graphIdentifier=args.graph_id)
        tasks = status_resp.get('tasks', [])
        # Find our task
        task = next((t for t in tasks if t['taskId'] == task_id), None)
        if not task:
            print("❌ Task not found; perhaps multiple tasks running? Check your graph ID.")
            break

        status = task['status']
        print(f"📊 Export Task {task_id} status: {status}")

        if status in ('SUCCEEDED', 'FAILED', 'CANCELLED'):
            print(f"🏁 Final status: {status}")
            print(f"📁 Destination: {task.get('destination')}")
            if status != 'SUCCEEDED':
                print(f"❌ Status reason: {task.get('statusReason')}")
            break

        time.sleep(15)  # wait before polling again

    if status == 'SUCCEEDED':
        # List exported files
        s3 = boto3.client('s3', region_name=args.region)
        resp = s3.list_objects_v2(Bucket=args.s3_bucket, Prefix=args.s3_prefix)
        files = [item['Key'] for item in resp.get('Contents', [])]
        print(f"\n📋 Exported files ({len(files)} total):")
        for f in files:
            print(f"  - {f}")
        
        print(f"\n✅ Export completed successfully!")
        print(f"📂 Files available at: s3://{args.s3_bucket}/{args.s3_prefix}")
    else:
        print(f"\n❌ Export failed with status: {status}")

if __name__ == "__main__":
    main()