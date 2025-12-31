# app.py (fast, deadline-safe)
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib, json, traceback
import pandas as pd
import numpy as np
from typing import List

# === paths ===
MODEL_PATH = "artifacts/model_pipeline.joblib"
LABEL_PATH = "artifacts/label_encoder.joblib"
FEATURES_PATH = "artifacts/feature_names.json"

# === load artifacts ===
pipeline = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_PATH)
with open(FEATURES_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)

# === find classifier inside pipeline (best-effort) ===
classifier = None
if hasattr(pipeline, "named_steps"):
    # last step is usually the estimator
    try:
        classifier = list(pipeline.named_steps.values())[-1]
    except Exception:
        classifier = pipeline
else:
    classifier = pipeline

# === create app + CORS ===
app = FastAPI(title="CyShield (fast)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "running", "docs": "/docs"}

def prepare_input(data: dict):
    """Create DataFrame with correct columns expected by the pipeline."""
    df = pd.DataFrame([data])
    # Add missing feature columns with 0
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
    # Drop extras and reorder
    df = df.loc[:, FEATURE_NAMES]
    return df

def quick_explanation(df: pd.DataFrame, pred_index: int) -> List[dict]:
    """
    Fast fallback explanation:
    - If classifier has feature_importances_ and length matches FEATURE_NAMES -> use that.
    - Otherwise return top input values as 'importance-like' (absolute value).
    """
    expl = []
    try:
        if hasattr(classifier, "feature_importances_"):
            fi = np.array(classifier.feature_importances_)
            if fi.shape[0] == len(FEATURE_NAMES):
                # top features by importance
                pairs = list(zip(FEATURE_NAMES, fi, df.iloc[0].tolist()))
                pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:6]
                for name, imp, val in pairs_sorted:
                    expl.append({"feature": name, "importance": float(imp), "value": float(val)})
                return expl
    except Exception:
        pass

    # fallback: pick top absolute values from the input data (fast)
    vals = np.abs(df.iloc[0].astype(float).fillna(0).values)
    pairs = list(zip(FEATURE_NAMES, vals))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)[:6]
    for name, v in pairs_sorted:
        expl.append({"feature": name, "importance": float(v), "value": float(df.iloc[0][name])})
    return expl

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        input_df = prepare_input(data)

        # Use the full pipeline (it may include preprocessing)
        probs = pipeline.predict_proba(input_df)[0]
        pred_index = int(np.argmax(probs))
        confidence = float(probs[pred_index])
        # decode label (pipeline predict_proba -> class index mapping)
        # We assume your label_encoder was saved as in training: it maps indices -> labels
        try:
            prediction = label_encoder.inverse_transform([pred_index])[0]
        except Exception:
            # fallback: pipeline.predict
            prediction = str(pipeline.predict(input_df)[0])

        # fast explanation
        explanation = quick_explanation(input_df, pred_index)

        return {"prediction": prediction, "confidence": round(confidence, 4), "explanation": explanation}

    except Exception as e:
        return {"error": "Prediction failed", "details": str(e), "trace": traceback.format_exc()}
