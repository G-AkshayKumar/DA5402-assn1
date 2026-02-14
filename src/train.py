import pandas as pd
import yaml
import os
import json
import joblib
import hashlib
from datetime import datetime
import subprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ---------------------------
# Utility Functions
# ---------------------------

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()
    except:
        return "NA"

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

# ---------------------------
# Training Pipeline
# ---------------------------

def main():
    config = load_config()

    version = config["data"]["current_version"]
    processed_dir = config["data"]["processed_dir"]

    train_file = os.path.join(processed_dir, f"{version}_train.csv")

    model_cfg = config["model"]
    registry_cfg = config["registry"]

    model_dir = registry_cfg["model_dir"]
    model_version = registry_cfg["current_model_version"]

    os.makedirs(model_dir, exist_ok=True)

    print("Loading processed training data...")
    df = pd.read_csv(train_file)

    target = "Machine failure"
    X = df.drop(columns=[target])
    y = df[target]

    print("Training model...")

    model = RandomForestClassifier(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
        random_state=model_cfg["random_state"]
    )

    model.fit(X, y)

    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)

    model_path = os.path.join(model_dir, f"{model_version}.pkl")
    joblib.dump(model, model_path)

    metadata = {
        "model_version": model_version,
        "training_date": datetime.now().isoformat(),
        "dataset_version": version,
        "git_commit": get_git_commit(),
        "algorithm": model_cfg["algorithm"],
        "hyperparameters": model_cfg,
        "metrics": {
            "accuracy": round(acc,4),
            "precision": round(prec,4),
            "recall": round(rec,4)
        },
        "model_hash": file_hash(model_path)
    }

    meta_file = os.path.join(model_dir, f"{model_version}_metadata.json")

    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=4)

    log_line = f"{model_version} | data:{version} | acc:{acc:.4f} | prec:{prec:.4f} | rec:{rec:.4f} | file:{model_path}\n"

    with open(os.path.join(model_dir, "model_metadata.log"), "a") as f:
        f.write(log_line)

    print("Training completed.")
    print("Saved model:", model_path)
    print("Saved metadata:", meta_file)

# ---------------------------
if __name__ == "__main__":
    main()
