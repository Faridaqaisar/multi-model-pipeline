import cv2
import numpy as np
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')


def stage1_detect_face(image_np):
    """Stage 1: Find a face in the image. Returns bounding box (x,y,w,h) or None."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    return faces[0]


def stage2_crop_face(image_np, box):
    """Stage 2: Crop the image to just the face region."""
    x, y, w, h = box
    return image_np[y:y+h, x:x+w]


def stage3_classify_smile(face_crop):
    """Stage 3: Detect whether the cropped face is smiling."""
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=22)
    return len(smiles) > 0


def run_pipeline(image_np):
    """Orchestrates all 3 stages, with timing and error handling per stage."""
    results = {"stages": []}

    t0 = time.time()
    box = stage1_detect_face(image_np)
    t1 = time.time()
    results["stages"].append({"name": "face_detection", "time_ms": round((t1-t0)*1000, 2), "success": box is not None})

    if box is None:
        results["error"] = "No face detected"
        return results

    results["bounding_box"] = [int(v) for v in box]

    t0 = time.time()
    face_crop = stage2_crop_face(image_np, box)
    t1 = time.time()
    results["stages"].append({"name": "crop", "time_ms": round((t1-t0)*1000, 2), "success": face_crop.size > 0})

    t0 = time.time()
    is_smiling = stage3_classify_smile(face_crop)
    t1 = time.time()
    results["stages"].append({"name": "smile_classification", "time_ms": round((t1-t0)*1000, 2), "success": True})

    results["label"] = "Smiling 😊" if is_smiling else "Not Smiling 😐"
    return results