#!/usr/bin/env python3
import pandas as pd
import os

DATA_BASE = "/home/ubuntu/scaling-law/data"
data_names = ["craft", "gender", "diabetes", "adult"]

for data_name in data_names:

    print(f"\n=== Creating seed_ind for {data_name} ===")

    use_path  = f"{DATA_BASE}/{data_name}/use.csv"
    seed_path = f"{DATA_BASE}/{data_name}/seed.csv"
    out_path  = f"{DATA_BASE}/{data_name}/seed_ind.csv"

    print(f"  Input  use.csv  : {use_path}")
    print(f"  Input  seed.csv : {seed_path}")
    print(f"  Output seed_ind.csv : {out_path}")

    use_df = pd.read_csv(use_path)
    seed_df = pd.read_csv(seed_path)

    # Drop seed rows based on full row match (outer merge)
    merged = pd.merge(use_df, seed_df, how="outer", indicator=True)
    seed_ind_df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

    seed_ind_df.to_csv(out_path, index=False)
    print(f"  Saved seed_ind.csv ({len(seed_ind_df)} rows)")
