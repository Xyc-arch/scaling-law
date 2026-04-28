import pandas as pd
import numpy as np
import json
import os
from utils import *

def split_and_save_ablate_syn(
    file_path,
    label_column="Target",
    spurious_column="X6",
    spurious_small_is_0=1,
    random_seed=42,
):
    # Base name will be "synthetic_seed"
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    parent_dir = os.path.dirname(file_path)

    # Output names derived from base name
    output_train1 = os.path.join(parent_dir, f"{base_name}1.csv")
    output_train0 = os.path.join(parent_dir, f"{base_name}0.csv")
    output_pos = os.path.join(parent_dir, f"{base_name}_pos.csv")
    output_neg = os.path.join(parent_dir, f"{base_name}_neg.csv")
    output_data_size = os.path.join(parent_dir, f"{base_name}_data_size.json")

    # Read synthetic data
    data = pd.read_csv(file_path)

    # Determine the small value for the spurious column
    if spurious_small_is_0 == 1:
        small_value = 0
    else:
        small_value = data[spurious_column].min()

    # Create spurious indicator
    data["spurious_value"] = data[spurious_column].apply(
        lambda x: 0 if x == small_value else 1
    )
    # Group by (spurious, label)
    data["group"] = (
        data["spurious_value"].astype(str) + data[label_column].astype(int).astype(str)
    )

    groups = {grp: df for grp, df in data.groupby("group")}
    for grp in ["00", "01", "10", "11"]:
        if grp not in groups:
            groups[grp] = pd.DataFrame(columns=data.columns)
            print(f"Warning: Group {grp} is empty.")

    # Build splits
    output_train1_data = (
        pd.concat([groups["01"], groups["11"]])
        .drop(["group", "spurious_value"], axis=1)
        .sample(frac=1, random_state=random_seed)
    )
    output_train0_data = (
        pd.concat([groups["00"], groups["10"]])
        .drop(["group", "spurious_value"], axis=1)
        .sample(frac=1, random_state=random_seed)
    )
    output_pos_data = (
        pd.concat([groups["00"], groups["11"]])
        .drop(["group", "spurious_value"], axis=1)
        .sample(frac=1, random_state=random_seed)
    )
    output_neg_data = (
        pd.concat([groups["01"], groups["10"]])
        .drop(["group", "spurious_value"], axis=1)
        .sample(frac=1, random_state=random_seed)
    )

    # Save outputs
    output_train1_data.to_csv(output_train1, index=False)
    output_train0_data.to_csv(output_train0, index=False)
    output_pos_data.to_csv(output_pos, index=False)
    output_neg_data.to_csv(output_neg, index=False)

    # Save counts
    data_size_info = {
        "train1_size": len(output_train1_data),
        "train0_size": len(output_train0_data),
        "pos_size": len(output_pos_data),
        "neg_size": len(output_neg_data),
    }
    with open(output_data_size, "w") as f:
        json.dump(data_size_info, f, indent=4)

    print(f"Finished ablation split for {file_path}")


if __name__ == "__main__":
    data_names = {0: "craft", 1: "gender", 2: "diabetes", 3: "adult"}
    data_name = data_names[3]  # choose dataset

    info = load_data_info("data_info.json")
    data_info = info.get(data_name, {})

    parent_path = "/home/ubuntu/scaling-law/"
    file_path = parent_path + f"/data/{data_name}/synthetic_seed.csv"

    split_and_save_ablate_syn(
        file_path,
        label_column=data_info.get("label_col"),
        spurious_column=data_info.get("spurious_col"),
        spurious_small_is_0=data_info.get("spurious_small_is_0"),
    )
