import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import streamlit as st
import time

# Load YOLOv8 model
model = YOLO('best.pt')

def process_frame(frame):
    # Run object detection
    results = model(frame)
    annotated_frame = results[0].plot()  # Plot the bounding boxes
    return annotated_frame

def main():
    st.title("YOLOv8 Object Detection")
    
    # File uploader for video
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    
    if video_file is not None:
        # Save the uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name
        
        # OpenCV video capture
        cap = cv2.VideoCapture(video_path)
        
        stframe = st.empty()  # Placeholder for the video stream
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get the original frame rate of the video
        frame_interval = int(fps // 5) 
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count +=1
            
            if frame_count % frame_interval != 0:
                continue
            
            # Process frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit
            annotated_frame = process_frame(frame)
            
            # Display frame
            stframe.image(annotated_frame, channels="RGB", use_column_width=True)
        
        cap.release()
    else:
        st.info("Please upload a video file to start detection.")

if __name__ == "__main__":
    main()























# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import tempfile
# import os


# # Title of the app
# st.title("Object Detection with YOLOv8")

# # Sidebar for uploading the video
# st.sidebar.header("Upload Video")
# uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

# # Load the YOLOv8 model
# st.sidebar.header("Model Settings")
# model_path = st.sidebar.text_input("Path to YOLOv8 Model", "best.pt")
# if st.sidebar.button("Load Model"):
#     model = YOLO(model_path)
#     st.sidebar.success("Model Loaded Successfully!")

# # Process video if a file is uploaded
# if uploaded_file is not None:
#     st.sidebar.info("Processing video...")

#     # Save the uploaded video to a temporary file
#     temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#     temp_video.write(uploaded_file.read())
#     video_path = temp_video.name

#     # Output video path
#     output_path = os.path.join(tempfile.gettempdir(), "output.mp4")

#     # OpenCV video processing
#     cap = cv2.VideoCapture(video_path)
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     # Process each frame
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         results = model(frame)
#         annotated_frame = results[0].plot()
#         out.write(annotated_frame)

#     cap.release()
#     out.release()

#     st.video(output_path)

# # Footer
# st.sidebar.header("About")
# st.sidebar.write("This app uses YOLOv8 for object detection on uploaded videos.")
