# ablate_ind_syn_split.py

import os
import json
import pandas as pd
from utils import load_data_info

ROOT      = "/home/ubuntu/scaling-law"
CODEBASE  = f"{ROOT}/codebase"
DATA_BASE = f"{ROOT}/data"
INFO_PATH = f"{CODEBASE}/data_info.json"

datasets = ["craft", "gender", "diabetes", "adult"]
info = load_data_info(INFO_PATH)

for data_name in datasets:
    print(f"\n=== Splitting IND synthetic data for: {data_name} ===")

    cfg = info[data_name]
    label_col    = cfg["label_col"]
    spurious_col = cfg["spurious_col"]
    small_is_0   = cfg["spurious_small_is_0"]

    file_path = f"{DATA_BASE}/{data_name}/synthetic_seed_ind.csv"
    parent    = f"{DATA_BASE}/{data_name}"

    df = pd.read_csv(file_path)

    # ---- spurious indicator ----
    if small_is_0 == 1:
        small_value = 0
    else:
        small_value = df[spurious_col].min()

    df["sp"] = df[spurious_col].apply(
        lambda x: 0 if x == small_value else 1
    )

    # IMPORTANT: cast label to int before string
    df["group"] = df["sp"].astype(str) + df[label_col].astype(int).astype(str)

    # collect 4 groups; drop helper cols
    groups = {}
    for g in ["00", "01", "10", "11"]:
        g_df = df[df["group"] == g].drop(columns=["sp", "group"])
        groups[g] = g_df

    # build splits
    g1   = pd.concat([groups["01"], groups["11"]], ignore_index=True)
    g0   = pd.concat([groups["00"], groups["10"]], ignore_index=True)
    gpos = pd.concat([groups["00"], groups["11"]], ignore_index=True)
    gneg = pd.concat([groups["01"], groups["10"]], ignore_index=True)

    # save
    g1.to_csv(f"{parent}/synthetic_seed_ind1.csv", index=False)
    g0.to_csv(f"{parent}/synthetic_seed_ind0.csv", index=False)
    gpos.to_csv(f"{parent}/synthetic_seed_ind_pos.csv", index=False)
    gneg.to_csv(f"{parent}/synthetic_seed_ind_neg.csv", index=False)

    # optional: sizes JSON for debugging
    size_info = {
        "g1_size":   len(g1),
        "g0_size":   len(g0),
        "pos_size":  len(gpos),
        "neg_size":  len(gneg),
    }
    with open(f"{parent}/synthetic_seed_ind_data_size.json", "w") as f:
        json.dump(size_info, f, indent=4)

    print("Done.")
    print(size_info)
