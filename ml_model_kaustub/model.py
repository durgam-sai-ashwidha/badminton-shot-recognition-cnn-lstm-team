import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CNNLSTM(nn.Module):

    def __init__(
        self,
        num_classes=5,
        hidden_size=256,
        num_layers=2,
        freeze_cnn=False,
        dropout=0.3
    ):

        super().__init__()

        weights = ResNet18_Weights.DEFAULT

        self.cnn = resnet18(weights=weights)

        self.cnn.fc = nn.Identity()

        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        self.feature_size = 512

        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        batch_size, num_frames, channels, height, width = x.shape

        x = x.view(
            batch_size * num_frames,
            channels,
            height,
            width
        )

        features = self.cnn(x)

        features = features.view(
            batch_size,
            num_frames,
            self.feature_size
        )

        lstm_output, _ = self.lstm(features)

        last_output = lstm_output[:, -1, :]

        last_output = self.dropout(last_output)

        output = self.fc(last_output)

        return output


if __name__ == "__main__":

    print("=" * 60)
    print("CNN + LSTM + DROPOUT MODEL TEST")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    model = CNNLSTM(
        num_classes=5,
        hidden_size=256,
        num_layers=2,
        freeze_cnn=False,
        dropout=0.3
    )

    model = model.to(device)

    x = torch.randn(
        2,
        16,
        3,
        224,
        224
    ).to(device)

    print(f"Input shape : {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")

    print("=" * 60)

    if output.shape == (2, 5):
        print("🎉 MODEL TEST PASSED!")
    else:
        print("⚠️ MODEL TEST FAILED!")

    print("=" * 60)