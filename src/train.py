import pandas as pd
import yaml
import os
import json
import joblib
import hashlib
from datetime import datetime
import subprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
    train_data_hash = file_hash(train_file)

    model_cfg = config["model"]
    registry_cfg = config["registry"]

    model_dir = registry_cfg["model_dir"]
    model_version = registry_cfg["current_model_version"]

    os.makedirs(model_dir, exist_ok=True)

    print("Loading processed training data...")
    df = pd.read_csv(train_file)

    target = config["data"]["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    print("Training model...")

    if model_cfg["algorithm"] == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=model_cfg["n_estimators"],
            max_depth=model_cfg["max_depth"],
            random_state=model_cfg["random_state"]
        )
    
    elif model_cfg["algorithm"] == "LogisticRegression":
        model = LogisticRegression(
            max_iter=1000,
            C=model_cfg.get("lr_C", 1.0)
        )


    else:
        raise ValueError("Unsupported algorithm in config.yaml")


    model.fit(X, y)
   # Evaluate on train
    train_preds = model.predict(X)
    train_acc = accuracy_score(y, train_preds)
    train_prec = precision_score(y, train_preds)
    train_rec = recall_score(y, train_preds)

    # Evaluate on test
    test_file = os.path.join(processed_dir, f"{version}_test.csv")
    test_df = pd.read_csv(test_file)

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds)
    test_rec = recall_score(y_test, test_preds)

    model_path = os.path.join(model_dir, f"{model_version}.pkl")
    joblib.dump(model, model_path)

    metadata = {
        "model_version": model_version,
        "training_date": datetime.now().isoformat(),
        "dataset_version": version,
        "training_data_hash": train_data_hash,
        "git_commit": get_git_commit(),
        "algorithm": model_cfg["algorithm"],
        "hyperparameters": model_cfg,
        "metrics": {
        "train": {
            "accuracy": round(train_acc,4),
            "precision": round(train_prec,4),
            "recall": round(train_rec,4)
        },
        "test": {
            "accuracy": round(test_acc,4),
            "precision": round(test_prec,4),
            "recall": round(test_rec,4)
        }
    },

        "model_hash": file_hash(model_path)
    }

    meta_file = os.path.join(model_dir, f"{model_version}_metadata.json")

    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=4)

    log_line = (
        f"{model_version} | algo:{model_cfg['algorithm']} | data:{version} | "
        f"train_acc:{train_acc:.4f} train_prec:{train_prec:.4f} train_rec:{train_rec:.4f} | "
        f"test_acc:{test_acc:.4f} test_prec:{test_prec:.4f} test_rec:{test_rec:.4f} | "
        f"file:{model_path}\n"
    )


    with open(os.path.join(model_dir, "model_metadata.log"), "a") as f:
        f.write(log_line)

    print("Training completed.")
    print("Saved model:", model_path)
    print("Saved metadata:", meta_file)

# ---------------------------
if __name__ == "__main__":
    main()
