import torch
from torch.utils.data import DataLoader, random_split
from dataset import BadmintonDataset
from model import CNNLSTM

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import os

# Resolve paths dynamically so the script runs from any directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR) if os.path.basename(SRC_DIR) == "src" else SRC_DIR
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_PATH = os.path.join(ROOT_DIR, "badminton_cnn_lstm.pth")

BATCH_SIZE = 4

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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = CNNLSTM(num_classes=5)
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for frames, labels in test_loader:

        frames = frames.to(device)
        labels = labels.to(device)

        outputs = model(frames)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision_weighted = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

precision_macro = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 50)
print("              MODEL EVALUATION RESULTS")
print("=" * 50)
print(f"Total Test Samples: {len(all_labels)}")
print(f"Overall Accuracy  : {accuracy * 100:.2f}%")
print("-" * 50)
print(f"Weighted Precision: {precision_weighted * 100:.2f}%")
print(f"Weighted Recall   : {recall_weighted * 100:.2f}%")
print(f"Weighted F1 Score : {f1_weighted * 100:.2f}%")
print("-" * 50)
print(f"Macro Precision   : {precision_macro * 100:.2f}%")
print(f"Macro Recall      : {recall_macro * 100:.2f}%")
print(f"Macro F1 Score    : {f1_macro * 100:.2f}%")

print("\n" + "=" * 55)
print("             PER-CLASS CLASSIFICATION REPORT")
print("=" * 55)

# IMPORTANT: Class names MUST match dataset.py label mapping:
# 0: Clear, 1: Drive, 2: Drop, 3: Net Shot, 4: Smash
class_names = [
    "Clear",
    "Drive",
    "Drop",
    "Net Shot",
    "Smash"
]

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

print("=" * 55)
print("                  CONFUSION MATRIX")
print("       (Rows: True Class | Columns: Predicted Class)")
print("=" * 55)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

col_header = f"{'True \\ Pred':<12} " + " ".join([f"{name[:8]:>8}" for name in class_names])
print(col_header)
print("-" * len(col_header))
for i, row in enumerate(cm):
    row_str = f"{class_names[i]:<12} " + " ".join([f"{val:>8d}" for val in row])
    print(row_str)
print("=" * 55)
