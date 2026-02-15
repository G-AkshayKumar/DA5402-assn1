import pandas as pd
import yaml
import joblib
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score

# -------------------------
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

# -------------------------
def main():
    config = load_config()

    model_path = config["deployment"]["model_path"]
    threshold = config["deployment"]["threshold"]

    prod_dir = config["data"]["production_dir"]
    prod_file = os.path.join(prod_dir, "production_day1.csv")

    model = joblib.load(model_path)

    df = pd.read_csv(prod_file)

    target = "Machine failure"

    X = df.drop(columns=[target])
    y = df[target]

    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)

    print("\n--- Production Metrics ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")

    meta_file = model_path.replace(".pkl", "_metadata.json")

    with open(meta_file) as f:
        meta = json.load(f)

    train_metrics = meta["metrics"]["test"]

    print("\n--- Training (Test) Metrics ---")
    print(train_metrics)

    acc_drop = acc < train_metrics["accuracy"] * threshold
    prec_drop = prec < train_metrics["precision"] * threshold
    rec_drop = rec < train_metrics["recall"] * threshold

    if acc_drop or prec_drop or rec_drop:
        print("\n⚠️ PERFORMANCE DROP DETECTED")
        print("Recommendation: Retrain model with new data.")
    else:
        print("\n✅ Model performance acceptable.")

# -------------------------
if __name__ == "__main__":
    main()
