# Badminton Shot Recognition (CNN-LSTM)

A Deep Learning application for recognizing badminton strokes from video clips using a CNN-LSTM architecture, FastAPI backend, and an interactive React + Vite frontend.

---

## 🏸 Shot Classes
The system classifies badminton video clips into 5 stroke categories:
1. **Clear**
2. **Drive**
3. **Drop**
4. **Net_Shot**
5. **Smash**

---

## 🏗️ Project Structure

```text
badminton-shot-recognition-cnn-lstm-team/
│
├── frontend/                     # React + Vite Interactive Dashboard
│   ├── public/                   # Static assets & icons
│   ├── src/                      # UI components, state management & styles
│   ├── package.json              # Frontend dependencies (React 19, Lucide, Vite)
│   └── vite.config.js            # Vite configuration
│
├── backend/                      # FastAPI Backend Service
│   └── main.py                   # REST API server (endpoints: /, /api/health, /predict)
│
├── models/                       # Team Models
│   ├── ml_model_ash/             # Ashwidha CNN-LSTM model & notebook
│   ├── ml_model_kaustub/         # Kaustub CNN-LSTM model
│   ├── ml_model_priyanshu/       # Priyanshu CNN-LSTM model
│   └── ml_model_rudra/           # Rudra CNN-LSTM model (Active Production Model)
│       ├── model/
│       │   └── model.pth         # Trained checkpoint (150 MB, Git LFS)
│       └── src/
│           ├── extract_frames.py # 16-frame uniform sampling & preprocessing
│           ├── model.py          # CNNLSTMPretrained (ResNet18 + 2-layer LSTM)
│           └── predict.py        # Video inference pipeline & class probabilities
│
├── .gitattributes                # Git LFS tracking for model weights
├── .gitignore                    # Git exclusions (node_modules, pycache, etc.)
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### 1. Start the Backend API
From the project root:
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- **Health Check**: `http://127.0.0.1:8000/api/health`
- **Swagger Documentation**: `http://127.0.0.1:8000/docs`

### 2. Start the Frontend
From the `frontend/` directory:
```bash
cd frontend
npm run dev
```
The application will be accessible at `http://localhost:5173`. Upload a video clip (`.mp4`, `.avi`, `.mov`, `.mkv`) and analyze the stroke prediction and confidence breakdown.
