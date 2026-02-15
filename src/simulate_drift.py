import pandas as pd
import yaml
import os
import numpy as np

# Loadinf Config

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

# Main pipeline

def main():
    config = load_config()

    version = config["data"]["current_version"]
    processed_dir = config["data"]["processed_dir"]
    prod_dir = config["data"]["production_dir"]

    os.makedirs(prod_dir, exist_ok=True)

    test_file = os.path.join(processed_dir, f"{version}_test.csv")
    df = pd.read_csv(test_file)

    # Introduce drift 
    target = config["data"]["target_column"]

    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns
    numeric_cols = [col for col in numeric_cols if col != target]

    for col in numeric_cols:
        if np.random.rand() < 0.25:
            df[col] = np.random.permutation(df[col].values)

    prod_file = os.path.join(prod_dir, "production_day2.csv")
    df.to_csv(prod_file, index=False)

    print("Drift simulated.")
    print("Saved:", prod_file)

if __name__ == "__main__":
    main()
