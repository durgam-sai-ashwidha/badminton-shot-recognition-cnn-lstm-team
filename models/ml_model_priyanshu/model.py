import torch
import torch.nn as nn
import torchvision.models as models


class CNNLSTM(nn.Module):

    def __init__(self, num_classes=5):
        super().__init__()

        cnn = models.resnet18(weights="DEFAULT")

        self.cnn = nn.Sequential(*list(cnn.children())[:-1])

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        batch_size, frames, channels, height, width = x.shape

        x = x.view(
            batch_size * frames,
            channels,
            height,
            width
        )

        features = self.cnn(x)

        features = features.view(
            batch_size,
            frames,
            512
        )

        output, _ = self.lstm(features)

        output = output[:, -1, :]

        output = self.fc(output)

        return output
