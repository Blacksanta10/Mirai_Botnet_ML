# Step 1: Purpose is to label, sample and clean the data

import pandas as pd
import numpy as np
import os

def final_load_data():
    benign = load_sampled_csv("data/benign.csv", label=0) # Check if this path is correct

    mirai = load_and_label("data/mirai", label=1)   # Check if this path is correct
    mirai = mirai.sample(n=10000, random_state=42)

    
    # Combine both samples from benign and mirai into one dataframe
    dataset = pd.concat([benign, mirai], ignore_index=True)

    X = dataset.drop("label", axis=1)
    y = dataset["label"]

    return X, y


# This will load and sample .CSV files
def load_sampled_csv(filepath, label, sample_size=10000):
    df = pd.read_csv(filepath)
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    df["label"] = label
    return df


def load_and_label(folder_path, label):
    dfs = []
    # Can load and label CSV files
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            df["label"] = label
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

