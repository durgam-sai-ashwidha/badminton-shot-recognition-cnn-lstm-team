"""
FastAPI Server for Badminton Shot Recognition.
Exposes Ashwidha CNN-LSTM model inference via REST API.
"""

import os
from pathlib import Path
import tempfile
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from sirish_backend.inference import predict_video
from sirish_backend.model import CLASS_NAMES

app = FastAPI(
    title="Badminton Shot Recognition API",
    description="Inference API using Ashwidha CNN-LSTM trained model.",
    version="1.0.0",
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    """
    Health check endpoint. Confirms service availability and model details.
    Does not run video inference.
    """
    return {
        "status": "ok",
        "model": "ashwidha_cnn_lstm",
        "classes": CLASS_NAMES,
    }


@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict_shot(video: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Receives an uploaded video clip, runs Ashwidha CNN-LSTM model inference,
    and returns predicted class, raw logits, and softmax probabilities.
    Temporary files are strictly removed after inference.
    """
    if not video.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No video file uploaded or filename is missing.",
        )

    # Validate file extension
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    suffix = Path(video.filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{suffix}'. Supported formats: {sorted(allowed_extensions)}",
        )

    temp_file_path = None
    try:
        # Save upload to a secure temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            temp_file_path = tmp_file.name
            content = await video.read()
            if not content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded video file is empty (0 bytes).",
                )
            tmp_file.write(content)

        # Run inference using existing sirish_backend.inference pipeline
        result = predict_video(temp_file_path)
        return {
            "predicted_index": result["predicted_index"],
            "predicted_class": result["predicted_class"],
            "probabilities": result["probabilities"],
            "logits": result["logits"],
        }

    except ValueError as ve:
        # Re-surface domain/validation errors from the notebook/inference layer
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )
    finally:
        # Guaranteed cleanup of temporary video file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


if __name__ == "__main__":
    uvicorn.run("sirish_backend.app:app", host="127.0.0.1", port=8000, reload=False)
