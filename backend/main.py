import io
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.vgg19 import preprocess_input

MODEL_PATH = os.getenv("MODEL_PATH", "vgg19_dos_detection.keras")
PREPROCESS_PATH = os.getenv("PREPROCESS_PATH", "vgg19_dos_preprocessing.joblib")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_ROWS = 50_000
BATCH_SIZE = 64
IMG_SIDE = 8
IMAGE_SIZE = 64

app = FastAPI(
    title="NetShield AI — VGG19 DoS Inference API",
    version="1.0.0",
    description="CIC-IDS2017 BENIGN vs DoS flow classification using the saved VGG19 model and preprocessing artifacts.",
)

origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

model = None
artifacts: dict[str, Any] | None = None


@app.on_event("startup")
def load_artifacts() -> None:
    global model, artifacts
    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    if not os.path.isfile(PREPROCESS_PATH):
        raise RuntimeError(f"Preprocessing artifact not found: {PREPROCESS_PATH}")
    artifacts = joblib.load(PREPROCESS_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)


@app.get("/")
def root():
    return {"service": "NetShield AI", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    ready = model is not None and artifacts is not None
    return {
        "status": "ok" if ready else "loading",
        "model": "VGG19 DoS binary classifier",
        "classes": ["BENIGN", "DoS"],
        "threshold": float(artifacts["threshold"]) if artifacts else None,
    }


def signed_log(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.log1p(np.abs(values))


def to_vgg_images(features: np.ndarray) -> tf.Tensor:
    images = tf.convert_to_tensor(features, dtype=tf.float32)
    images = tf.reshape(images, (-1, IMG_SIDE, IMG_SIDE, 1)) * 255.0
    images = tf.image.resize(images, (IMAGE_SIZE, IMAGE_SIZE), method="nearest")
    images = tf.repeat(images, 3, axis=-1)
    return preprocess_input(images)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or artifacts is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV exceeds the 25 MB upload limit")
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty")

    try:
        frame = pd.read_csv(io.BytesIO(raw), low_memory=False, encoding="latin1")
        frame.columns = frame.columns.astype(str).str.strip()
        if frame.empty:
            raise HTTPException(status_code=400, detail="The CSV contains no data rows")
        if len(frame) > MAX_ROWS:
            raise HTTPException(status_code=413, detail=f"CSV has {len(frame):,} rows; limit is {MAX_ROWS:,}")
        if "Label" in frame.columns:
            frame = frame.drop(columns=["Label"])

        required = list(artifacts["keep_columns"])
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=(f"CSV is missing {len(missing)} required model features. "
                        f"First missing columns: {missing[:10]}"),
            )

        data = frame[required].apply(pd.to_numeric, errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(artifacts["medians"])

        values = signed_log(data.to_numpy())
        values = np.clip(artifacts["minmax"].transform(values), 0.0, 1.0)
        values = artifacts["selector"].transform(values)[:, artifacts["order"]]
        target_features = int(artifacts["n_features"])
        if values.shape[1] < target_features:
            values = np.hstack(
                [values, np.zeros((len(values), target_features - values.shape[1]))]
            )
        values = values.astype("float32")

        probabilities: list[float] = []
        for start in range(0, len(values), BATCH_SIZE):
            batch_images = to_vgg_images(values[start : start + BATCH_SIZE])
            batch_probabilities = model.predict(batch_images, verbose=0).reshape(-1)
            probabilities.extend(float(p) for p in batch_probabilities)

        threshold = float(artifacts["threshold"])
        labels = ["DoS" if probability >= threshold else "BENIGN" for probability in probabilities]
        predictions = [
            {
                "row": index + 1,
                "label": labels[index],
                "dos_probability": probabilities[index],
                "confidence": max(probabilities[index], 1.0 - probabilities[index]),
            }
            for index in range(len(labels))
        ]
        return {
            "total_flows": len(predictions),
            "dos_count": labels.count("DoS"),
            "benign_count": labels.count("BENIGN"),
            "threshold": threshold,
            "predictions": predictions,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not process CSV: {error}") from error
