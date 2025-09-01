import os, io, hashlib, argparse
from datetime import datetime
import pandas as pd
import boto3

def norm_str(x):
    if pd.isna(x): return None
    s = str(x).strip()
    return s if s and s.lower() != "nan" else None

def to_date(x):
    if pd.isna(x) or x is None: return None
    try:
        return pd.to_datetime(x).date().isoformat()
    except Exception:
        return None

def safe_content_id(row):
    # Prefer cntt_id; else stable hash from url/cntt_nm/tactic+act
    if pd.notna(row.get("cntt_id")) and str(row["cntt_id"]).strip():
        return f"Content#{str(row['cntt_id']).strip()}"
    key = row.get("url") or row.get("cntt_nm") or f"{row.get('act_nm','')}_{row.get('tact_id','')}"
    hid = hashlib.sha1((key or 'UNKNOWN').encode("utf-8")).hexdigest()[:16]
    return f"Content#h{hid}"

def edge_id(prefix, src, dst, extra=None):
    base = f"{prefix}|{src}|{dst}"
    if extra is not None:
        base += f"|{extra}"
    return base

def transform_to_neptune_csv(input_csv: str) -> dict:
    df = pd.read_csv(input_csv)

    # Normalize
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(norm_str)
    df["act_cnt"] = pd.to_numeric(df.get("act_cnt"), errors="coerce").fillna(0).astype(int)
    df["tact_id"] = pd.to_numeric(df.get("tact_id"), errors="coerce").astype("Int64")
    df["cpgn_id"] = pd.to_numeric(df.get("cpgn_id"), errors="coerce").astype("Int64")
    df["act_dt_iso"] = df["act_dt"].apply(to_date)

    outputs = {}

    # -------- Vertices --------
    hcp = df[["npi","email"]].dropna(subset=["npi"]).drop_duplicates().copy()
    hcp["~id"] = "HCP#" + hcp["npi"].astype(str)
    hcp["~label"] = "HCP"
    hcp.rename(columns={"npi":"npi:String","email":"email:String"}, inplace=True)
    hcp = hcp[["~id","~label","npi:String","email:String"]]
    outputs["vertices_hcp.csv"] = hcp

    brand = df[["brd"]].dropna().drop_duplicates().copy()
    brand["~id"] = "Brand#" + brand["brd"]; brand["~label"] = "Brand"
    brand.rename(columns={"brd":"brd:String"}, inplace=True)
    brand = brand[["~id","~label","brd:String"]]
    outputs["vertices_brand.csv"] = brand

    camp = df[["cpgn","cpgn_id"]].dropna(subset=["cpgn"]).drop_duplicates().copy()
    camp["~id"] = "Campaign#" + camp["cpgn"]; camp["~label"] = "Campaign"
    camp.rename(columns={"cpgn":"cpgn:String","cpgn_id":"cpgn_id:Int"}, inplace=True)
    camp = camp[["~id","~label","cpgn:String","cpgn_id:Int"]]
    outputs["vertices_campaign.csv"] = camp

    tact = df[["tact_id"]].dropna(subset=["tact_id"]).drop_duplicates().copy()
    tact["~id"] = "Tactic#" + tact["tact_id"].astype(str); tact["~label"] = "Tactic"
    tact.rename(columns={"tact_id":"tact_id:Int"}, inplace=True)
    tact = tact[["~id","~label","tact_id:Int"]]
    outputs["vertices_tactic.csv"] = tact

    chan = df[["chnl"]].dropna().drop_duplicates().copy()
    chan["~id"] = "Channel#" + chan["chnl"]; chan["~label"] = "Channel"
    chan.rename(columns={"chnl":"chnl:String"}, inplace=True)
    chan = chan[["~id","~label","chnl:String"]]
    outputs["vertices_channel.csv"] = chan

    srcv = df[["src"]].dropna().drop_duplicates().copy()
    srcv["~id"] = "Source#" + srcv["src"]; srcv["~label"] = "Source"
    srcv.rename(columns={"src":"src:String"}, inplace=True)
    srcv = srcv[["~id","~label","src:String"]]
    outputs["vertices_source.csv"] = srcv

    ta = df[["ta"]].dropna().drop_duplicates().copy()
    ta["~id"] = "TherapyArea#" + ta["ta"]; ta["~label"] = "TherapyArea"
    ta.rename(columns={"ta":"ta:String"}, inplace=True)
    ta = ta[["~id","~label","ta:String"]]
    outputs["vertices_therapy_area.csv"] = ta

    ind = df[["indc"]].dropna().drop_duplicates().copy()
    ind["~id"] = "Indication#" + ind["indc"]; ind["~label"] = "Indication"
    ind.rename(columns={"indc":"indc:String"}, inplace=True)
    ind = ind[["~id","~label","indc:String"]]
    outputs["vertices_indication.csv"] = ind

    cnt = df[["cntt_id","cntt_nm","url"]].copy()
    cnt["~id"] = df.apply(safe_content_id, axis=1); cnt["~label"] = "Content"
    cnt = cnt.drop_duplicates(subset=["~id"])
    cnt.rename(columns={"cntt_id":"cntt_id:String","cntt_nm":"cntt_nm:String","url":"url:String"}, inplace=True)
    cnt = cnt[["~id","~label","cntt_id:String","cntt_nm:String","url:String"]]
    outputs["vertices_content.csv"] = cnt

    # -------- Edges --------
    e_bc = df[["brd","cpgn"]].dropna().drop_duplicates().copy()
    e_bc["~from"] = "Brand#" + e_bc["brd"]; e_bc["~to"] = "Campaign#" + e_bc["cpgn"]
    e_bc["~id"] = [edge_id("HAS_CAMPAIGN", f, t) for f,t in zip(e_bc["~from"], e_bc["~to"])]
    e_bc["~label"] = "HAS_CAMPAIGN"
    e_bc = e_bc[["~id","~from","~to","~label"]]
    outputs["edges_brand_has_campaign.csv"] = e_bc

    e_ct = df[["cpgn","tact_id"]].dropna().drop_duplicates().copy()
    e_ct["~from"] = "Campaign#" + e_ct["cpgn"]; e_ct["~to"] = "Tactic#" + e_ct["tact_id"].astype(str)
    e_ct["~id"] = [edge_id("USES_TACTIC", f, t) for f,t in zip(e_ct["~from"], e_ct["~to"])]
    e_ct["~label"] = "USES_TACTIC"
    e_ct = e_ct[["~id","~from","~to","~label"]]
    outputs["edges_campaign_uses_tactic.csv"] = e_ct

    e_tr = df[["tact_id","chnl"]].dropna().drop_duplicates().copy()
    e_tr["~from"] = "Tactic#" + e_tr["tact_id"].astype(str); e_tr["~to"] = "Channel#" + e_tr["chnl"]
    e_tr["~id"] = [edge_id("RUNS_ON", f, t) for f,t in zip(e_tr["~from"], e_tr["~to"])]
    e_tr["~label"] = "RUNS_ON"
    e_tr = e_tr[["~id","~from","~to","~label"]]
    outputs["edges_tactic_runs_on_channel.csv"] = e_tr

    e_ts = df[["tact_id","src"]].dropna().drop_duplicates().copy()
    e_ts["~from"] = "Tactic#" + e_ts["tact_id"].astype(str); e_ts["~to"] = "Source#" + e_ts["src"]
    e_ts["~id"] = [edge_id("SOURCED_FROM", f, t) for f,t in zip(e_ts["~from"], e_ts["~to"])]
    e_ts["~label"] = "SOURCED_FROM"
    e_ts = e_ts[["~id","~from","~to","~label"]]
    outputs["edges_tactic_sourced_from.csv"] = e_ts

    e_tp = df[["tact_id","cntt_id","cntt_nm","url"]].copy()
    e_tp["to_id"] = df.apply(safe_content_id, axis=1)
    e_tp = e_tp.dropna(subset=["tact_id","to_id"]).drop_duplicates(subset=["tact_id","to_id"])
    e_tp["~from"] = "Tactic#" + e_tp["tact_id"].astype(str)
    e_tp["~to"] = e_tp["to_id"]
    e_tp["~id"] = [edge_id("PROMOTES", f, t) for f,t in zip(e_tp["~from"], e_tp["~to"])]
    e_tp["~label"] = "PROMOTES"
    e_tp = e_tp[["~id","~from","~to","~label"]]
    outputs["edges_tactic_promotes_content.csv"] = e_tp

    e_cta = df[["cpgn","ta"]].dropna().drop_duplicates().copy()
    e_cta["~from"] = "Campaign#" + e_cta["cpgn"]; e_cta["~to"] = "TherapyArea#" + e_cta["ta"]
    e_cta["~id"] = [edge_id("TARGETS_TA", f, t) for f,t in zip(e_cta["~from"], e_cta["~to"])]
    e_cta["~label"] = "TARGETS_TA"
    e_cta = e_cta[["~id","~from","~to","~label"]]
    outputs["edges_campaign_targets_ta.csv"] = e_cta

    e_ci = df[["cpgn","indc"]].dropna().drop_duplicates().copy()
    e_ci["~from"] = "Campaign#" + e_ci["cpgn"]; e_ci["~to"] = "Indication#" + e_ci["indc"]
    e_ci["~id"] = [edge_id("TARGETS_INDICATION", f, t) for f,t in zip(e_ci["~from"], e_ci["~to"])]
    e_ci["~label"] = "TARGETS_INDICATION"
    e_ci = e_ci[["~id","~from","~to","~label"]]
    outputs["edges_campaign_targets_indication.csv"] = e_ci

    e_eng = df[["id","npi","tact_id","act_nm","act_dt_iso","act_cnt","url","src_act_nm","src_chnl_nm","devc_nm"]]\
            .dropna(subset=["npi","tact_id"]).copy()
    e_eng["~from"] = "HCP#" + e_eng["npi"].astype(str)
    e_eng["~to"]   = "Tactic#" + e_eng["tact_id"].astype(str)
    e_eng["~id"]   = e_eng.apply(lambda r: edge_id("ENGAGED", r["~from"], r["~to"], str(r["id"])), axis=1)
    e_eng["~label"] = "ENGAGED"
    e_eng.rename(columns={
        "act_nm":"act_nm:String",
        "act_dt_iso":"act_dt:Date",
        "act_cnt":"act_cnt:Int",
        "url":"url:String",
        "src_act_nm":"src_act_nm:String",
        "src_chnl_nm":"src_chnl_nm:String",
        "devc_nm":"devc_nm:String"
    }, inplace=True)
    e_eng = e_eng[["~id","~from","~to","~label","act_nm:String","act_dt:Date","act_cnt:Int","url:String",
                   "src_act_nm:String","src_chnl_nm:String","devc_nm:String"]]
    outputs["edges_hcp_engaged_tactic.csv"] = e_eng

    return outputs

def upload_frames_to_s3(dfs: dict, bucket: str, prefix: str, s3_client) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base = f"{prefix.rstrip('/')}/{ts}/"
    for name, frame in dfs.items():
        buf = io.StringIO()
        frame.to_csv(buf, index=False)
        s3_client.put_object(Bucket=bucket, Key=base + name, Body=buf.getvalue())
    return f"s3://{bucket}/{base}"

def main():
    parser = argparse.ArgumentParser(description="Load HCP campaign data into Neptune Analytics")
    parser.add_argument("--region", required=True, help="AWS region")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket name")
    parser.add_argument("--s3-prefix", required=True, help="S3 prefix for data files")
    parser.add_argument("--role-arn", required=True, help="IAM role ARN for Neptune import")
    parser.add_argument("--graph-id", required=True, help="Neptune Analytics graph ID")
    parser.add_argument("--input-csv", default="./data/10_HCP_Sample_Data_Digital_Cpgn_Act.csv", 
                       help="Input CSV file path")
    
    args = parser.parse_args()
    
    # Initialize AWS clients
    session = boto3.session.Session(region_name=args.region)
    s3 = session.client("s3")
    ng = session.client("neptune-graph")
    
    frames = transform_to_neptune_csv(args.input_csv)
    s3_path = upload_frames_to_s3(frames, args.s3_bucket, args.s3_prefix, s3)
    print("Uploaded Neptune CSV files to:", s3_path)

    resp = ng.start_import_task(
        graphIdentifier=args.graph_id,
        source=s3_path,
        format="CSV",     # CSV (Gremlin)
        roleArn=args.role_arn,
        failOnError=False
    )
    print("Import started:")
    print("  taskId :", resp["taskId"])
    print("  status :", resp["status"])

if __name__ == "__main__":
    main()