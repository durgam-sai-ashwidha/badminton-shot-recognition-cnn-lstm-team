import os
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR) if os.path.basename(SRC_DIR) == "src" else SRC_DIR

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_PATH = os.path.join(ROOT_DIR, "badminton_cnn_lstm.pth")

CLASSES = ["clear", "drive", "drop", "net", "smash"]
CLASS_NAMES = ["Clear", "Drive", "Drop", "Net Shot", "Smash"]
NUM_CLASSES = len(CLASSES)

NUM_FRAMES = 16
IMAGE_SIZE = 224

BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

