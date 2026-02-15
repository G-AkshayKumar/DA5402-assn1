import joblib
import yaml
import pandas as pd
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import subprocess

# Load Config

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
    except:
        return "NA"

config = load_config()

MODEL_PATH = config["deployment"]["model_path"]
PORT = config["deployment"]["port"]
MODEL_VERSION = os.path.basename(MODEL_PATH)

# Load Model

try:
    model = joblib.load(MODEL_PATH)
except:
    raise RuntimeError("Model file not found. Check config.yaml")

# FastAPI App

app = FastAPI(title="Predictive Maintenance API")

class InputData(BaseModel):
    features: list

# Deployment Logging

def log_deployment():
    if not os.path.exists("deployment_log.csv"):
        with open("deployment_log.csv","w") as f:
            f.write("timestamp,model_file,model_version,git_commit,port,notes\n")

    timestamp = datetime.now().isoformat()
    git_commit = get_git_commit()

    row = f"{timestamp},{MODEL_PATH},{MODEL_VERSION},{git_commit},{PORT},manual_deploy\n"

    with open("deployment_log.csv", "a") as f:
        f.write(row)

log_deployment()

# Routes

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: InputData):
    EXPECTED_FEATURES = model.n_features_in_

    if len(data.features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_FEATURES} features, got {len(data.features)}"
        )

    try:
        df = pd.DataFrame([data.features], columns=model.feature_names_in_)
        prob = model.predict_proba(df)[0][1]
        pred = int(prob >= 0.5)

        return {
            "prediction": pred,
            "probability": round(float(prob),4),
            "model_version": MODEL_VERSION,
            "model_file" : MODEL_PATH
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
