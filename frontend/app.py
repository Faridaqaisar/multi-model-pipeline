import streamlit as st
import requests
import base64
from PIL import Image
import io

st.set_page_config(page_title="Multi-Model Pipeline", layout="wide")
st.title("🔍 Face Detection + Smile Classification Pipeline")
st.write("Upload a photo with a face. It will move through 3 stages: detect → crop → classify.")
st.info("ℹ️ This demo's backend runs locally. For a live working demo, clone the repo and run both backend and frontend together — see the GitHub README for instructions.")

API_URL = "http://127.0.0.1:8000"

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.subheader("Original Image")
    st.image(uploaded_file, width=300)

    if st.button("Run Pipeline"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        with st.spinner("Processing through pipeline stages..."):
            try:
                response = requests.post(f"{API_URL}/pipeline", files=files, timeout=5)
                result = response.json()
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Backend is not running. This demo requires the FastAPI backend to be running locally alongside this app. See the GitHub README for setup instructions.")
                st.stop()

        if "error" in result:
            st.error(f"Stage failed: {result['error']}")
        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Stage 1: Detection")
                annotated_bytes = base64.b64decode(result["annotated_image_base64"])
                annotated_img = Image.open(io.BytesIO(annotated_bytes))
                st.image(annotated_img, caption="Face detected (green box)", use_container_width=True)

            with col2:
                st.subheader("Stage 2: Crop")
                cropped_bytes = base64.b64decode(result["cropped_face_base64"])
                cropped_img = Image.open(io.BytesIO(cropped_bytes))
                st.image(cropped_img, caption="Cropped to face region", use_container_width=True)

            with col3:
                st.subheader("Stage 3: Classification")
                st.markdown(f"## {result['label']}")

            st.divider()
            st.subheader("Pipeline Timing (per stage)")
            for stage in result["stages"]:
                status = "✅" if stage["success"] else "❌"
                st.write(f"{status} **{stage['name']}**: {stage['time_ms']} ms")