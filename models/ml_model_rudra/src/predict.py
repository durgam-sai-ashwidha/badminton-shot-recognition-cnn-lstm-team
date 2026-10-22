
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from src.model import CNNLSTMPretrained


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "model.pth"

CLASS_NAMES = ["Clear", "Drive", "Drop", "Net_Shot", "Smash"]
NUM_FRAMES = 16
IMAGE_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def extract_frames(video_path):
    """Extract 16 evenly spaced frames from a video."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("The video contains no readable frames.")

    indices = np.linspace(
        0, total_frames - 1, NUM_FRAMES, dtype=int
    )

    frames = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = cap.read()

        if not success:
            cap.release()
            raise ValueError(f"Could not read frame {index}.")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        frames.append(TRANSFORM(image))

    cap.release()

    # Shape: [1, 16, 3, 224, 224]
    return torch.stack(frames).unsqueeze(0)


def load_model():
    """Load the trained checkpoint."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = CNNLSTMPretrained()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    # Support either a raw state_dict or a checkpoint dictionary.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def predict_video(video_path):
    """Return the predicted badminton stroke and class probabilities."""
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = load_model()
    video_tensor = extract_frames(video_path).to(DEVICE)

    with torch.no_grad():
        logits = model(video_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": round(
            float(probabilities[predicted_index].item()), 4
        ),
        "probabilities": {
            name: round(float(probabilities[i].item()), 4)
            for i, name in enumerate(CLASS_NAMES)
        },
    }


if __name__ == "__main__":
    test_video = ROOT_DIR / "data" / "raw" / "test.avi"

    result = predict_video(test_video)
    print("Prediction:", result["predicted_class"])
    print("Confidence:", result["confidence"])
    print("Probabilities:", result["probabilities"])
