
import torch
import torch.nn as nn
from torchvision import models


class CNNLSTMPretrained(nn.Module):
    def __init__(
        self,
        num_classes=5,
        hidden_size=256,
        num_layers=2
    ):
        super().__init__()

        # CNN: extract spatial features from each frame
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Identity()

        # LSTM: learn temporal patterns across video frames
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.4
        )

        # Fully connected layer for stroke classification
        # Name "fc" matches your checkpoint's fc.weight and fc.bias
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Expected input: [batch, frames, channels, height, width]
        batch_size, seq_len, channels, height, width = x.shape

        # Process each frame through ResNet18
        x = x.reshape(
            batch_size * seq_len,
            channels,
            height,
            width
        )

        x = self.cnn(x)

        # Restore the video sequence
        x = x.reshape(batch_size, seq_len, 512)

        # Process the sequence using the LSTM
        x, _ = self.lstm(x)

        # Use the final time step
        x = x[:, -1, :]

        # Return raw class scores (logits)
        return self.fc(x)
