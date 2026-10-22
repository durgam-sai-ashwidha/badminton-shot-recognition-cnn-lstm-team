🏸 Badminton CNN-LSTM



An AI-powered badminton stroke recognition system that uses a CNN-LSTM deep learning model to classify badminton shots from video.



The project combines a React frontend, FastAPI backend, OpenCV video processing, and a PyTorch CNN-LSTM model to provide an end-to-end video-based shot prediction system.






📌 Overview



Badminton involves fast and complex movements where identifying individual strokes from match footage can be difficult to perform manually.



This project aims to automate badminton stroke recognition by analyzing video frames and classifying the detected stroke into one of five categories:



\- Clear

\- Drive

\- Drop Shot

\- Net Shot

\- Smash



The system accepts a badminton video through a web interface, processes the video, extracts a sequence of frames, and passes the frames through a trained CNN-LSTM model.



The final prediction and confidence score are returned to the React frontend.



🎯 Objective



The main objective of this project is to build an automated badminton stroke recognition system that can:



\- Accept badminton video as input

\- Extract representative frames from the video

\- Learn spatial information from individual frames

\- Learn temporal information across consecutive frames

\- Classify the badminton stroke

\- Return the predicted class and confidence

\- Provide the result through a web-based interface






🧠 System Architecture



> An architecture diagram will be added here.



🧠 System Architecture

The overall system follows this pipeline:

```text
User
  ↓
React Frontend
  ↓
HTTP POST Request
  ↓
FastAPI Backend
  ↓
Video Upload
  ↓
predict.py
  ↓
OpenCV Video Processing
  ↓
16 Sampled Frames
  ↓
CNN Spatial Feature Extraction
  ↓
LSTM Temporal Modeling
  ↓
5-Class Classification
  ↓
Softmax Probabilities
  ↓
JSON Response
  ↓
React Frontend
  ↓
Prediction + Confidence
```

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
