import os
import sys
import cv2
import torch
from torchvision import models
from model import CNNLSTM

CLASSES = ["Clear", "Drive", "Drop", "Net Shot", "Smash"]
NUM_FRAMES = 16

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR) if os.path.basename(SRC_DIR) == "src" else SRC_DIR
MODEL_PATH = os.path.join(ROOT_DIR, "badminton_cnn_lstm.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    sys.exit(1)

try:
    model = CNNLSTM(num_classes=5)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model = model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

if len(sys.argv) > 1:
    video_path = sys.argv[1].strip().strip('"')
else:
    video_path = input("Enter video path: ").strip().strip('"')

if not os.path.exists(video_path):
    print(f"Error: Video file not found at '{video_path}'")
    sys.exit(1)

video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print(f"Error: Unable to open video file '{video_path}'. Invalid or unsupported format.")
    sys.exit(1)

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames < NUM_FRAMES:
    print(f"Error: Video has only {total_frames} frames. At least {NUM_FRAMES} frames are required.")
    video.release()
    sys.exit(1)

frames = []

for i in range(NUM_FRAMES):
    frame_number = int(i * total_frames / NUM_FRAMES)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = video.read()

    if not success or frame is None:
        print(f"Error: Failed to extract frame {frame_number} from '{video_path}'.")
        video.release()
        sys.exit(1)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = torch.tensor(frame, dtype=torch.float32)
    frame = frame.permute(2, 0, 1)
    frames.append(frame)

video.release()

if len(frames) != NUM_FRAMES:
    print(f"Error: Expected {NUM_FRAMES} frames, but extracted {len(frames)}.")
    sys.exit(1)

frames = torch.stack(frames)
frames = frames.unsqueeze(0)
frames = frames.to(device)

with torch.no_grad():
    output = model(frames)
    probabilities = torch.softmax(output, dim=1)[0]
    prediction = torch.argmax(probabilities).item()
    confidence = probabilities[prediction].item() * 100

video_name = os.path.basename(video_path)

print("\n" + "=" * 40)
print("       BADMINTON SHOT RECOGNITION")
print("=" * 40)
print(f"Video          : {video_name}")
print(f"Input Tensor   : {list(frames.shape)}")
print(f"Predicted Shot : {CLASSES[prediction].upper()}")
print(f"Confidence     : {confidence:.2f}%")
print("-" * 40)
print("Class Probabilities:")
for i, cls_name in enumerate(CLASSES):
    print(f"  {cls_name:<10} : {probabilities[i].item() * 100:6.2f}%")
print("=" * 40 + "\n")
