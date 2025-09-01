import boto3, io, argparse

def main():
    parser = argparse.ArgumentParser(description="Verify HCP campaign data loaded into Neptune Analytics")
    parser.add_argument("--region", required=True, help="AWS region")
    parser.add_argument("--graph-id", required=True, help="Neptune Analytics graph ID")
    
    args = parser.parse_args()
    
    # Initialize Neptune client
    ng = boto3.client("neptune-graph", region_name=args.region)
    
    print("🔍 Verifying Neptune Analytics data load...")
    print(f"Graph ID: {args.graph_id}")
    print(f"Region: {args.region}")
    print("-" * 50)
    
    # Query 1: Sample HCP nodes
    print("📋 Sample HCP nodes:")
    res = ng.execute_query(
        graphIdentifier=args.graph_id, 
        language="OPEN_CYPHER",
        queryString="MATCH (h:HCP) RETURN h LIMIT 5"
    )
    print(res["payload"].read().decode("utf-8"))
    
    # Query 2: Node counts by type
    print("\n📊 Node counts by type:")
    res = ng.execute_query(
        graphIdentifier=args.graph_id, 
        language="OPEN_CYPHER",
        queryString="MATCH (n) RETURN labels(n) AS label, count(n) AS num_nodes;"
    )
    print(res["payload"].read().decode("utf-8"))
    
    # Query 3: Total edge count
    print("\n🔗 Total edge count:")
    res = ng.execute_query(
        graphIdentifier=args.graph_id, 
        language="OPEN_CYPHER",
        queryString="MATCH ()-[r]->() RETURN count(r) AS total_edges;"
    )
    print(res["payload"].read().decode("utf-8"))
    
    # Query 4: Sample engagement relationships
    print("\n🎯 Sample HCP engagement relationships:")
    res = ng.execute_query(
        graphIdentifier=args.graph_id, 
        language="OPEN_CYPHER",
        queryString="MATCH (h:HCP)-[e:ENGAGED]->(t:Tactic) RETURN h.npi, t.tact_id, e.act_cnt LIMIT 5"
    )
    print(res["payload"].read().decode("utf-8"))
    
    print("\n✅ Data verification complete!")

if __name__ == "__main__":
    main()
