import pandas as pd
import yaml
import joblib
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Loading Config
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

# Main pipeline
def main():
    config = load_config()

    model_path = config["deployment"]["model_path"]
    threshold = config["deployment"]["threshold"]

    prod_dir = config["data"]["production_dir"]
    prod_file = os.path.join(prod_dir, "production_day2.csv")

    target = config["data"]["target_column"]

    model = joblib.load(model_path)
    model_name = os.path.basename(model_path)

    df = pd.read_csv(prod_file)

    X = df.drop(columns=[target])
    y = df[target]

    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)

    prod_error = 1 - acc

    print("\n--- Production Metrics ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")

    print("\n--- Error Rates ---")
    print(f"Production Error Rate : {prod_error:.4f}")

    meta_file = model_path.replace(".pkl", "_metadata.json")

    with open(meta_file) as f:
        meta = json.load(f)

    train_metrics = meta["metrics"]["test"]
    train_error = 1 - train_metrics["accuracy"]

    print("\n--- Deployed Model Info ---")
    print(f"Model File   : {model_name}")
    print(f"Model Version: {meta['model_version']}")
    print(f"Algorithm    : {meta['algorithm']}")

    print("\n--- Training (Test) Metrics ---")
    print(train_metrics)

    print("\n--- Error Rates (Baseline) ---")
    print(f"Training Test Error : {train_error:.4f}")

    print("\n--- Threshold Check ---")
    print(f"Accuracy threshold  : {train_metrics['accuracy'] * threshold:.4f}")
    print(f"Precision threshold : {train_metrics['precision'] * threshold:.4f}")
    print(f"Recall threshold    : {train_metrics['recall'] * threshold:.4f}")

    acc_drop = acc < train_metrics["accuracy"] * threshold
    prec_drop = prec < train_metrics["precision"] * threshold
    rec_drop = rec < train_metrics["recall"] * threshold

    error_threshold = train_error * (1 / threshold)
    error_drop = prod_error > error_threshold

    if acc_drop or prec_drop or rec_drop or error_drop:
        print("\n PERFORMANCE DROP DETECTED")
        print("Recommendation: Retrain model with new data.")
    else:
        print("\n Model performance acceptable.")

if __name__ == "__main__":
    main()
