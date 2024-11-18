import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os


# Title of the app
st.title("Object Detection with YOLOv8")

# Sidebar for uploading the video
st.sidebar.header("Upload Video")
uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

# Load the YOLOv8 model
st.sidebar.header("Model Settings")
model_path = st.sidebar.text_input("Path to YOLOv8 Model", "best.pt")
if st.sidebar.button("Load Model"):
    model = YOLO(model_path)
    st.sidebar.success("Model Loaded Successfully!")

# Process video if a file is uploaded
if uploaded_file is not None:
    st.sidebar.info("Processing video...")

    # Save the uploaded video to a temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video.write(uploaded_file.read())
    video_path = temp_video.name

    # Output video path
    output_path = os.path.join(tempfile.gettempdir(), "output.mp4")

    # OpenCV video processing
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    st.video(output_path)

# Footer
st.sidebar.header("About")
st.sidebar.write("This app uses YOLOv8 for object detection on uploaded videos.")
