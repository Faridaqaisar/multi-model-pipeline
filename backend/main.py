import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pipeline import run_pipeline, stage2_crop_face

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def image_to_base64(image_np):
    success, buffer = cv2.imencode('.jpg', image_np)
    return base64.b64encode(buffer).decode('utf-8')


@app.post("/pipeline")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image_np is None:
        return {"error": "Could not read image file"}

    results = run_pipeline(image_np)

    if "bounding_box" in results:
        annotated = image_np.copy()
        x, y, w, h = results["bounding_box"]
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 3)
        results["annotated_image_base64"] = image_to_base64(annotated)

        face_crop = stage2_crop_face(image_np, results["bounding_box"])
        results["cropped_face_base64"] = image_to_base64(face_crop)

    return results


@app.get("/")
def root():
    return {"message": "Multi-model pipeline API running. POST an image to /pipeline"}