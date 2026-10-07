"""
Ashwidha CNN-LSTM Model Architecture and Loading Utilities.
Badminton Shot Recognition.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models import resnet18

# Verified class names and index mappings from Ashwidha notebook
CLASS_NAMES: List[str] = [
    "Smash",
    "Clear",
    "Net Shot",
    "Drop",
    "Drive",
]

CLASS_TO_IDX: Dict[str, int] = {
    "Smash": 0,
    "Clear": 1,
    "Net Shot": 2,
    "Drop": 3,
    "Drive": 4,
}

IDX_TO_CLASS: Dict[int, str] = {
    0: "Smash",
    1: "Clear",
    2: "Net Shot",
    3: "Drop",
    4: "Drive",
}

DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent
    / "ml_model_ash"
    / "best_badminton_cnn_lstm_model2.pth"
)


class BadmintonCNNLSTM(nn.Module):
    """
    CNN-LSTM deep learning architecture for Badminton Shot Recognition.
    Backbone: ResNet18 with fc replaced by Identity.
    Temporal: 2-layer LSTM with hidden size 256.
    Classifier: Dropout(0.5) -> Linear(256, 5).
    """

    def __init__(
        self,
        num_classes: int = 5,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Pretrained ResNet18 backbone
        self.cnn = resnet18(weights=None)
        self.cnn.fc = nn.Identity()

        # LSTM sequence modeling
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, seq_len, C=3, H=224, W=224)
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, C, H, W = x.size()

        # Combine batch and temporal dimensions for CNN feature extraction
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.cnn(x)

        # Restore temporal sequence dimension
        features = features.view(batch_size, seq_len, -1)

        # Temporal sequence modeling through LSTM
        lstm_out, _ = self.lstm(features)

        # Aggregate using the last frame's LSTM hidden state
        last_output = lstm_out[:, -1, :]

        # Output logits
        output = self.classifier(last_output)
        return output


def get_default_device(prefer_cuda: bool = False) -> torch.device:
    """
    Returns torch.device. Defaults to CPU for safe portability.
    If prefer_cuda is True and CUDA is available, returns cuda device.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_badminton_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    prefer_cuda: bool = False,
) -> Tuple[BadmintonCNNLSTM, Dict]:
    """
    Instantiates BadmintonCNNLSTM, loads weights from the checkpoint,
    sets model to eval mode, and returns (model, checkpoint_metadata).
    """
    if checkpoint_path is None:
        target_path = DEFAULT_CHECKPOINT_PATH
    else:
        target_path = Path(checkpoint_path)

    if not target_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {target_path}")

    if device is None:
        device = get_default_device(prefer_cuda=prefer_cuda)

    model = BadmintonCNNLSTM(num_classes=5, hidden_size=256, num_layers=2)

    checkpoint = torch.load(target_path, map_location=device, weights_only=True)

    load_result = model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    metadata = {
        "epoch": checkpoint.get("epoch"),
        "val_accuracy": checkpoint.get("val_accuracy"),
        "val_f1": checkpoint.get("val_f1"),
        "device": str(device),
        "checkpoint_path": str(target_path),
        "load_result": str(load_result),
    }

    return model, metadata
