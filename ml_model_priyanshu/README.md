# Enhancing Badminton Performance Analytics with CNN-LSTM Shot Recognition

## 1. Project Objective
This project implements an end-to-end deep learning framework to automatically classify badminton strokes from video clips. By fusing spatial feature extraction (CNN) with temporal sequence modeling (LSTM), the system enables automated shot recognition for coaching feedback, performance analytics, and player match statistics.

---

## 2. Five Badminton Shot Classes
The system strictly recognizes **5 core badminton strokes**:
- **0: Clear** — High, deep shot directed to the opponent's backcourt.
- **1: Drive** — Flat, fast horizontal exchange travelling close across the net.
- **2: Drop** — Soft overhead stroke descending steeply into the opponent's frontcourt.
- **3: Net Shot** — Delicate touch stroke played close to the net from the forecourt.
- **4: Smash** — High-velocity, steep downward overhead attacking stroke.

*(No extraneous classes such as Lift are included).*

---

## 3. CNN-LSTM Hybrid Architecture
- **Spatial Feature Extractor (CNN)**: ResNet18 pre-trained convolutional neural network. The final classification head is replaced by an identity mapping, producing a **512-dimensional feature embedding** for each video frame.
- **Temporal Sequence Modeling (LSTM)**: 2-layer Long Short-Term Memory network with:
  - `input_size = 512`
  - `hidden_size = 256`
  - `batch_first = True`
- **Classification Head**: Fully connected `Linear(256, 5)` mapping the final hidden state to 5 class logits, followed by Softmax activation.
- **Active Model**: `C:\batminton\badminton_cnn_lstm.pth` (50 MB)
- **Original V1 Fallback Backup**: `C:\batminton\badminton_cnn_lstm_backup.pth` (50 MB)

---

## 4. Input Video Format & Preprocessing
- **Sequence Length**: Exactly **16 frames** uniformly sampled across the stroke duration.
- **Spatial Resolution**: **224 × 224 pixels**.
- **Color Format**: **RGB** (converted from OpenCV's BGR).
- **Pixel Normalization**: Scaled to `[0.0, 1.0]` by dividing pixel values by `255.0`.
- **Input Tensor Dimensions**: `[Batch, Frames, Channels, Height, Width]` = `[1, 16, 3, 224, 224]`.

---

## 5. End-to-End Pipeline Architecture
```
[ Input Video (.mp4) ]
         │
         ▼
[ 16 Uniformly Sampled Frames (OpenCV) ]
         │
         ▼
[ Preprocessing: BGR->RGB, 224x224, /255.0 ]
         │
         ▼
[ Tensor: 1 x 16 x 3 x 224 x 224 (CUDA / CPU) ]
         │
         ▼
[ ResNet18 CNN: 16 x 512 Feature Vectors ]
         │
         ▼
[ 2-Layer LSTM (Hidden Size: 256) ]
         │
         ▼
[ Linear Layer (256 -> 5) + Softmax ]
         │
         ▼
[ Predicted Shot & Confidence Probabilities ]
```

---

## 6. How to Run Prediction

### A. Direct Video Path (Command-Line)
```bash
python src/predict.py "path/to/video.mp4"
```

### B. Interactive Mode
```bash
python src/predict.py
# Prompt will ask: Enter video path:
```

---

## 7. Example Prediction Output

```text
========================================
       BADMINTON SHOT RECOGNITION
========================================
Video          : CLEAR_MATCH02_SHOT0001_HITFRAME00012016.mp4
Input Tensor   : [1, 16, 3, 224, 224]
Predicted Shot : CLEAR
Confidence     : 98.56%
----------------------------------------
Class Probabilities:
  Clear      :  98.56%
  Drive      :   0.08%
  Drop       :   1.06%
  Net Shot   :   0.19%
  Smash      :   0.10%
========================================
```

---

## 8. Verified Empirical Model Results

| Benchmark Dataset | Sample Count | V1 Baseline Model | Final Promoted Model | Generalization Note |
| :--- | :---: | :---: | :---: | :--- |
| **Original V1 Holdout Test** | 204 clips | **96.57%** (197/204) | **93.14%** (190/204) | High benchmark fidelity retained |
| **PART 2 Held-out Test** | 366 clips | 43.72% (160/366) | **83.33%** (305/366) | Multi-match cross-arena test (+39.61%) |
| **Combined Test Pool** | 570 clips | 62.63% (357/570) | **86.84%** (495/570) | Overall test accuracy (+24.21%) |
| **Genuinely Unseen Sample Clips** | 13 clips | 46.15% (6/13) | **84.62%** (11/13) | +38.46% generalization gain |

The final model was promoted because it solved the cross-match domain shift observed in V1, successfully distinguishing overhead trajectories (Clear vs Drop vs Smash) across diverse lighting and court environments.
