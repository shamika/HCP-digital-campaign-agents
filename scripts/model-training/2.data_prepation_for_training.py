# prepare_and_upload.py
import argparse, io
import boto3
import pandas as pd

def list_s3(bucket, prefix, contains=None):
    s3 = boto3.client("s3")
    token = None
    out = []
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token: kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for c in resp.get("Contents", []):
            k = c["Key"]
            if (contains is None) or (contains in k):
                out.append(k)
        token = resp.get("NextContinuationToken")
        if not token: break
    return sorted(out)

def read_csvs(bucket, keys):
    s3 = boto3.client("s3")
    dfs = []
    for k in keys:
        obj = s3.get_object(Bucket=bucket, Key=k)
        dfs.append(pd.read_csv(io.BytesIO(obj["Body"].read())))
    if not dfs:
        raise RuntimeError("No CSVs found for given pattern.")
    return pd.concat(dfs, ignore_index=True)

def write_csv_s3(df, bucket, key):
    s3 = boto3.client("s3")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"wrote s3://{bucket}/{key}  rows={len(df)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--export-prefix", required=True,
                    help="e.g. neptune/hcp_marketing-exported/t-6ka1g84iga/")
    ap.add_argument("--processed-prefix", required=True,
                    help="where to write processed files, e.g. graph-processed/hcp/")
    args = ap.parse_args()

    bucket = args.bucket
    base = args.export_prefix.rstrip("/") + "/"

    # 1) collect shards
    v_hcp_keys    = list_s3(bucket, base, contains="Vertex_HCP_")
    v_tactic_keys = list_s3(bucket, base, contains="Vertex_Tactic_")
    e_eng_keys    = list_s3(bucket, base, contains="Edge_ENGAGED_")

    if not v_hcp_keys:    raise SystemExit("No Vertex_HCP_* files found")
    if not v_tactic_keys: raise SystemExit("No Vertex_Tactic_* files found")
    if not e_eng_keys:    raise SystemExit("No Edge_ENGAGED_* files found")

    # 2) concat
    v_hcp    = read_csvs(bucket, v_hcp_keys)
    v_tactic = read_csvs(bucket, v_tactic_keys)
    e_eng    = read_csvs(bucket, e_eng_keys)

    # Expect Gremlin CSV headers (~id/~label on vertices; ~from/~to/~label on edges)
    # Neptune Analytics CSV exports follow Gremlin CSV conventions. :contentReference[oaicite:0]{index=0}

    # 3) minimal column sanity (keep system columns + all props)
    assert "~id" in v_hcp.columns and "~label" in v_hcp.columns
    assert "~id" in v_tactic.columns and "~label" in v_tactic.columns
    assert {"~from","~to","~label"}.issubset(set(e_eng.columns))

    # 4) write processed
    proc = args.processed_prefix.rstrip("/") + "/"
    write_csv_s3(v_hcp,    bucket, proc + "vertices_hcp.csv")
    write_csv_s3(v_tactic, bucket, proc + "vertices_tactic.csv")
    write_csv_s3(e_eng,    bucket, proc + "edges_engaged.csv")

    print("Done. Use these files as inputs to training.")
