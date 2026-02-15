import pandas as pd
import numpy as np
import yaml
import os
import hashlib
from datetime import datetime
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------------
# Utility Functions
# -------------------------------

np.random.seed(42)

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def sha256_checksum(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()
    except:
        return "NA"

# -------------------------------
# Main Pipeline
# -------------------------------

def main():
    config = load_config()

    raw_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]
    version = config["data"]["current_version"]

    test_size = config["preprocessing"]["test_size"]
    random_state = config["preprocessing"]["random_state"]

    os.makedirs(processed_dir, exist_ok=True)

    print("Loading raw dataset...")
    df = pd.read_csv(raw_path)

    # -------------------------------
    # Basic Cleaning
    # -------------------------------

    print("Cleaning data...")

    # Drop duplicates
    df = df.drop_duplicates()

    # Fill numeric missing values with median
    target_column = config["data"]["target_column"]
    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns
    numeric_cols = [c for c in numeric_cols if c != target_column]

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Encode categorical columns
    if config["preprocessing"]["encode_categorical"]:
        cat_cols = df.select_dtypes(include=["object","string"]).columns
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col])

    # -------------------------------
    # Train-Test Split
    # -------------------------------

    target_column = "Machine failure"

    X = df.drop(columns=[target_column])
    y = df[target_column]

    split_index = int(len(df) * (1 - test_size))

    X_train = X.iloc[:split_index]
    X_test  = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test  = y.iloc[split_index:]



    # -------------------------------
    # Scaling
    # -------------------------------

    if config["preprocessing"]["scale_numeric"]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df[target_column] = y_train.values

    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df[target_column] = y_test.values

    train_file = os.path.join(processed_dir, f"{version}_train.csv")
    test_file = os.path.join(processed_dir, f"{version}_test.csv")

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    # -------------------------------
    # Manifest Logging
    # -------------------------------

    train_hash = sha256_checksum(train_file)
    test_hash = sha256_checksum(test_file)
    git_commit = get_git_commit()

    timestamp = datetime.now().isoformat()
    
    manifest_entry = f"""
===============================
VERSION: {version}
DATE: {timestamp}
RAW_DATA: {raw_path}
SCRIPT: src/data_prep.py
GIT_COMMIT: {git_commit}

FILES:
- {train_file} | sha256: {train_hash}
- {test_file}  | sha256: {test_hash}

NOTES:
- Dropped duplicates
- Median imputation for numerics
- Label encoded categoricals
- Standard scaling applied
- Train/Test split

===============================
"""

    with open("manifest.txt", "a") as f:
        f.write(manifest_entry)

    print("Phase A completed successfully.")
    print(f"Saved: {train_file}")
    print(f"Saved: {test_file}")
    print("Manifest updated.")

# -------------------------------
if __name__ == "__main__":
    main()
