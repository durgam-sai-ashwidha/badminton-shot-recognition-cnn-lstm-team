"""
Video Preprocessing and Inference Pipeline for Ashwidha CNN-LSTM Model.
Badminton Shot Recognition.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from sirish_backend.model import (
    CLASS_NAMES,
    IDX_TO_CLASS,
    BadmintonCNNLSTM,
    load_badminton_model,
)

# Preprocessing transforms exactly as defined in the Ashwidha notebook
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_video(
    video_path: Union[str, Path],
    num_frames: int = 16,
) -> torch.Tensor:
    """
    Preprocess a video file exactly following Ashwidha's notebook logic:
    - Sample exactly 16 evenly spaced frames via np.linspace(0, total_frames - 1, 16)
    - Convert BGR to RGB
    - Resize to (224, 224)
    - ToTensor() and ImageNet normalization
    - On failed read (ret is False), substitute a black frame (224x224x3) through the same transform
    - Returns a tensor of shape (1, 16, 3, 224, 224) with dtype torch.float32

    Raises ValueError if video is corrupt, empty, or has fewer than 16 frames,
    as behavior for those cases is not defined by the existing notebook.
    """
    path_str = str(video_path)
    cap = cv2.VideoCapture(path_str)

    if not cap.isOpened():
        cap.release()
        raise ValueError(
            f"Cannot open video at '{path_str}' (empty or corrupt). "
            "Behavior for corrupt or unreadable videos is not specified in the existing notebook — we must not invent behavior yet."
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Notebook requires selecting 16 evenly spaced frames
    if total_frames < num_frames:
        cap.release()
        raise ValueError(
            f"Video has {total_frames} frames, which is fewer than the required {num_frames} frames. "
            "Behavior for videos shorter than 16 frames is not specified in the existing notebook — we must not invent behavior yet."
        )

    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()

        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tensor = TRANSFORM(img)
            frames.append(img_tensor)
        else:
            # Replicating notebook fallback for failed frame read
            black_frame = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            black_tensor = TRANSFORM(black_frame)
            frames.append(black_tensor)

    cap.release()

    # Stack into [16, 3, 224, 224] and add batch dimension -> [1, 16, 3, 224, 224]
    video_tensor = torch.stack(frames, dim=0).unsqueeze(0).to(dtype=torch.float32)
    return video_tensor


def predict_video(
    video_path: Union[str, Path],
    model: Optional[BadmintonCNNLSTM] = None,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Runs inference on an input video using Ashwidha's trained CNN-LSTM model.

    Returns:
        dict with:
            - predicted_index (int): argmax index over logits
            - predicted_class (str): class name mapped from predicted_index
            - logits (list[float]): raw unnormalized model output logits
            - probabilities (dict[str, float]): separate softmax probability values
              (clearly distinguished from notebook's raw argmax logic)
    """
    if model is None:
        model, _ = load_badminton_model(checkpoint_path=checkpoint_path, device=device)

    target_device = next(model.parameters()).device
    model.eval()

    # Preprocess video tensor
    tensor = preprocess_video(video_path, num_frames=16).to(target_device)

    with torch.no_grad():
        logits_tensor = model(tensor)  # Shape: (1, 5)

    # Argmax calculation as defined in the notebook evaluation loops
    predicted_index = int(logits_tensor.argmax(dim=1).item())
    predicted_class = IDX_TO_CLASS[predicted_index]
    logits = logits_tensor.squeeze(0).cpu().tolist()

    # Softmax probabilities kept separate from original notebook logic
    probs_tensor = torch.softmax(logits_tensor, dim=1)
    probabilities = probs_tensor.squeeze(0).cpu().tolist()
    class_probabilities = {
        CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_index": predicted_index,
        "predicted_class": predicted_class,
        "logits": logits,
        "probabilities": class_probabilities,
    }
