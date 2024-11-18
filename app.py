import cv2
import numpy as np
import tempfile
import pandas as pd
from ultralytics import YOLO
import streamlit as st
import altair as alt

# Load YOLOv8 model
model = YOLO('best.pt')

def detect_animals(image):
    """Detect animals in the given image and return the annotated image and counts."""
    animal_counts = {}
    results = model(image)
    annotated_image = results[0].plot()  # Plot bounding boxes
    for result in results:
        for class_index in result.boxes.cls:
            class_name = model.names[int(class_index)]  # Map class index to class name
            animal_counts[class_name] = animal_counts.get(class_name, 0) + 1
    return annotated_image, animal_counts

def process_video_frame(video_path):
    """Process video frame by frame and yield annotated frames and counts."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(fps // 5)  # Process 1 frame every 5th of a second

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame, animal_counts = detect_animals(frame)
        yield annotated_frame, animal_counts
    cap.release()

def plot_animal_counts(animal_counts):
    """Generate a bar chart for animal counts."""
    if animal_counts:
        data = pd.DataFrame(list(animal_counts.items()), columns=["Animal", "Count"])
        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x="Animal:O",
                y="Count:Q",
                color="Animal:O",
                tooltip=["Animal", "Count"]
            )
            .properties(width=400, height=300)
        )
        return chart
    else:
        return None

def main():
    st.title("Species Identification and Monitoring Terrestrial")

    # Sidebar for user input
    st.sidebar.header("Upload Options")
    upload_type = st.sidebar.radio("Select Input Type", ["Video", "Image"])

    if upload_type == "Video":
        video_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
        if video_file:
            # Save video file to a temporary location
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            video_path = tfile.name

            st.sidebar.header("Detection Progress")
            progress_bar = st.sidebar.progress(0)

            # Main layout: video display and chart
            video_placeholder = st.empty()  # Placeholder for the video
            chart_placeholder = st.empty()  # Placeholder for the chart below video

            fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)  # Get frame rate
            total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

            # Process and display video frame by frame
            for idx, (annotated_frame, animal_counts) in enumerate(process_video_frame(video_path)):
                # Update progress bar
                progress_bar.progress((idx + 1) / total_frames)

                # Display the video frame in a specific window size
                video_placeholder.image(
                    annotated_frame, 
                    channels="RGB", 
                    use_column_width=False, 
                    width=640
                )

            # Display the chart after the video
            chart = plot_animal_counts(animal_counts)
            if chart:
                chart_placeholder.altair_chart(chart)

    elif upload_type == "Image":
        image_file = st.sidebar.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
        if image_file:
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            annotated_image, animal_counts = detect_animals(image)

            st.image(annotated_image, channels="RGB", use_column_width=True, caption="Detected Animals")

            chart = plot_animal_counts(animal_counts)
            if chart:
                st.altair_chart(chart)

if __name__ == "__main__":
    main()



#------------------------------------------------------------------------
# import cv2
# import numpy as np
# import tempfile
# from ultralytics import YOLO
# import streamlit as st

# # Load YOLOv8 model
# model = YOLO('best.pt')

# def process_frame(frame):
#     # Run object detection
#     results = model(frame)
#     annotated_frame = results[0].plot()  # Plot the bounding boxes
#     return annotated_frame

# def main():
#     st.title("YOLOv8 Object Detection")
    
#     # File uploader for video
#     video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    
#     if video_file is not None:
#         # Save the uploaded file temporarily
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(video_file.read())
#         video_path = tfile.name
        
#         # OpenCV video capture
#         cap = cv2.VideoCapture(video_path)
        
#         stframe = st.empty()  # Placeholder for the video stream
#         fps = cap.get(cv2.CAP_PROP_FPS)  # Get the original frame rate of the video
#         frame_interval = int(fps // 5) 
        
#         frame_count = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame_count +=1
            
#             if frame_count % frame_interval != 0:
#                 continue
            
#             # Process frame
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit
#             annotated_frame = process_frame(frame)
            
#             # Display frame
#             stframe.image(annotated_frame, channels="RGB", use_column_width=True)
        
#         cap.release()
#     else:
#         st.info("Please upload a video file to start detection.")

# if __name__ == "__main__":
#     main()

#----------------------------------------------------------------------------------------------

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
