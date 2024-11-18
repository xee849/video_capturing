from ultralytics import YOLO
import numpy as np
import tempfile
import streamlit as st
import cv2

# Load YOLOv8 model
model = YOLO('best.pt')

def process_frame(frame):
    # Run object detection
    results = model(frame)
    annotated_frame = results[0].plot()  # Plot the bounding boxes
    return annotated_frame

def main():
    st.title("Species Identification and Monitoring Terrestrial")
    
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