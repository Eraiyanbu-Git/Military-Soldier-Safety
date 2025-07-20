import streamlit as st
import tempfile
import os
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

# Set page config
st.set_page_config(page_title="Military Object Detection", layout="centered")

# Title
st.title("🪖 Military Soldier Safety & Weapon Detection using YOLOv8")

# Load model once and cache it
@st.cache_resource
def load_model():
    model_path = "/Users/eraiyanbu/Desktop/Projects/Military/best.pt"  # Make sure this path exists
    return YOLO(model_path)

model = load_model()

# Sidebar for input selection
option = st.sidebar.selectbox("Choose input type", ["Image", "Video"])

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Save uploaded image to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            img.save(tmp_file.name)
            tmp_path = tmp_file.name

        st.success("Running detection...")

        # Run YOLO prediction
        results = model.predict(source=tmp_path, conf=0.25, save=True)

        # Display results
        result_img_path = Path(results[0].save_dir) / os.path.basename(tmp_path)
        if result_img_path.exists():
            st.image(str(result_img_path), caption="Detected Image", use_column_width=True)
        else:
            st.error("Result image not found.")

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_video.read())
            video_path = tfile.name

        st.video(video_path)
        st.success("Running detection on video...")

        # Run YOLO prediction
        results = model.predict(source=video_path, conf=0.25, save=True)

        # Display result video
        result_vid_path = Path(results[0].save_dir) / os.path.basename(video_path)
        if result_vid_path.exists():
            st.video(str(result_vid_path))
        else:
            st.warning("Detection complete, but video preview unavailable. Check output folder.")
