import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import BadmintonDataset
from model import CNNLSTM

DATA_DIR = "data/processed"

BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

dataset = BadmintonDataset(DATA_DIR)

print("Total samples:", len(dataset))

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = CNNLSTM(num_classes=5)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for frames, labels in train_loader:

        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(frames)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f}"
    )

torch.save(model.state_dict(), "badminton_cnn_lstm.pth")

print("Training completed!")
print("Model saved as badminton_cnn_lstm.pth")
