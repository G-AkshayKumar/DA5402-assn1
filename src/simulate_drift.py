import pandas as pd
import yaml
import os
import numpy as np

# -------------------------
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

# -------------------------
def main():
    config = load_config()

    version = config["data"]["current_version"]
    processed_dir = config["data"]["processed_dir"]
    prod_dir = config["data"]["production_dir"]

    os.makedirs(prod_dir, exist_ok=True)

    test_file = os.path.join(processed_dir, f"{version}_test.csv")
    df = pd.read_csv(test_file)

    # Introduce drift by shifting numeric values
    target = "Machine failure"

    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns
    numeric_cols = [col for col in numeric_cols if col != target]

    for col in numeric_cols:
        df[col] = df[col] * np.random.uniform(1.1, 1.3)


    prod_file = os.path.join(prod_dir, "production_day1.csv")
    df.to_csv(prod_file, index=False)

    print("Drift simulated.")
    print("Saved:", prod_file)

# -------------------------
if __name__ == "__main__":
    main()
