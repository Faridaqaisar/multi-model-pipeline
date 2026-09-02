# 🔍 Multi-Model Pipeline: Face Detection + Smile Classification

An image processing pipeline that chains multiple models together, similar to real-world computer vision systems (e.g., detect → crop → classify). Each stage's output feeds directly into the next, with per-stage timing and error handling.

🔗 **Live Demo (UI only):** [multi-model-pipeline.streamlit.app](https://multi-model-pipeline.streamlit.app)

> **Note:** The live demo shows the interface, but the backend runs locally for this project (not deployed to the cloud). To see the full working pipeline, follow the local setup instructions below.

## Tech Stack
FastAPI, OpenCV (Haar Cascade classifiers), Streamlit

## Pipeline Stages
1. **Face Detection** — locates a face in the uploaded image, returns a bounding box
2. **Crop** — crops the image to just the detected face region
3. **Smile Classification** — analyzes the cropped face to determine if the person is smiling

## Features
- Visual, multi-stage UI: shows the original image with a bounding box, the cropped face, and the final classification
- Per-stage timing breakdown
- Graceful error handling when no face is detected

## Running Locally (Full Working Demo)

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend (in a second terminal):**
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`, upload a clear front-facing photo, and click "Run Pipeline."

## Project Structure
multi-model-pipeline/
├── backend/
│ ├── main.py
│ ├── pipeline.py
│ └── requirements.txt
└── frontend/
└── app.py

## Note on Dependencies
This project pins `opencv-python==4.10.0.84` — newer releases (5.0+) were found to have a packaging bug where `cv2.CascadeClassifier` is missing.

## Why This Design
Real-world AI systems often chain specialized models together (e.g., security cameras: detect → crop → classify → OCR). This project demonstrates that orchestration pattern, along with the UI challenge of visualizing multi-stage state rather than a single input/output pair.
