import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
from dotenv import load_dotenv

from utils.emotions import get_encouraging_sentence, standardize_emotion_label

load_dotenv()


API_BASE_URL = os.getenv('API__URL')
st.set_page_config(page_title="Emotion Detection", page_icon="🎭", initial_sidebar_state="expanded")


st.sidebar.markdown('# 😎 MoodMate')
st.sidebar.markdown('---')
st.sidebar.page_link('Home.py', label='Home')
st.sidebar.page_link('pages/1_📘_Type_your_thoughts.py', icon='🖋️')
st.sidebar.page_link('pages/2_🎙️_Speak_your_thoughts.py', icon='🎙️')
st.sidebar.page_link('pages/3_🙂_Express_your_thoughts.py', icon='🧑')
st.sidebar.page_link('pages/4_🏃‍♂️_Recommendations_for_you.py', icon='🏃‍♂️')
st.sidebar.markdown('---')

# Set up temporary directory for video uploads
TEMP_DIR = Path("temp_videos")
TEMP_DIR.mkdir(exist_ok=True)

# Streamlit configuration


def stream_video_feed():
    """Stream the video feed from the Flask API and display it on Streamlit."""
    video_container = st.empty()  # Create a placeholder for the video feed
    
    try:
        # Send a GET request to the Flask `/video_feed` endpoint
        response = requests.get(f"{API_BASE_URL}/video_feed", stream=True)

        # Initialize a byte accumulator
        byte_accumulator = b''

        # Read the stream from the response
        for chunk in response.iter_content(chunk_size=1024):
            byte_accumulator += chunk

            # Detect frame boundary (end of the image data)
            start = byte_accumulator.find(b'\xff\xd8')  # JPEG start
            end = byte_accumulator.find(b'\xff\xd9')  # JPEG end

            if start != -1 and end != -1:
                # Extract a complete JPEG frame
                jpg_data = byte_accumulator[start:end + 2]
                byte_accumulator = byte_accumulator[end + 2:]  # Remove processed frame

                # Convert to a NumPy array and decode
                frame = cv2.imdecode(np.frombuffer(jpg_data, np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to PIL image
                    frame_pil = Image.fromarray(frame_rgb)
                    # Display in Streamlit
                    video_container.image(frame_pil, use_column_width=True)
    except requests.RequestException as e:
        st.error(f"Failed to connect to the webcam feed: {e}")


def stop_webcam():
    """Send a request to stop the webcam and display the generalized emotion result."""
    try:
        response = requests.post(f"{API_BASE_URL}/stop")
        if response.status_code == 200:
            emotion = standardize_emotion_label(response.text)
            s = get_encouraging_sentence(emotion)
            st.info(s)
            st.page_link('pages/4_🏃‍♂️_Recommendations_for_you.py', icon='👉')
            
            # st.success(f"**Generalized Emotion Detected:** {emotion}")
            st.session_state['video_last_clf'] = emotion
        else:
            st.error("Failed to stop the webcam or retrieve emotion data.")
    except requests.RequestException as e:
        st.error(f"Failed to send stop request: {e}")


def save_uploaded_file(uploaded_file):
    """Save the uploaded file to a temporary directory."""
    temp_file_path = TEMP_DIR / uploaded_file.name
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    return temp_file_path


def upload_and_detect_emotion(uploaded_file):
    """Upload the video file to the server for emotion detection."""
    temp_file_path = save_uploaded_file(uploaded_file)

    try:
        # Send the file to the Flask backend
        with open(temp_file_path, "rb") as f:
            files = {"file": (uploaded_file.name, f, "multipart/form-data")}
            response = requests.post(f"{API_BASE_URL}/upload_video", files=files)

        if response.status_code == 200:
            # Display the generalized emotion result
            generalized_emotion = standardize_emotion_label(response.text)
            s = get_encouraging_sentence(generalized_emotion)
            st.info(s)
            st.page_link('pages/4_🏃‍♂️_Recommendations_for_you.py', icon='👉')
            
            # st.success(f"**Generalized Emotion Detected:** {generalized_emotion}")
            st.session_state['video_last_clf'] = generalized_emotion
        else:
            st.error("Failed to process the video. Please try again.")
    except requests.RequestException as e:
        st.error(f"Failed to upload the video: {e}")
    finally:
        # Remove the temporary file after processing
        os.remove(temp_file_path)


# Streamlit UI for Webcam Emotion Detection
st.title("📷 Express your thoughts")
st.markdown(" --- ")
st.markdown("### Webcam Emotion Detection")
st.markdown("Use your webcam to detect emotions in real-time.")

# Webcam Start/Stop Controls
left, right = st.columns(2)

if left.button("Start Webcam Detection", use_container_width=True, type='primary'):
    stream_video_feed()

if right.button("Stop Webcam Detection", use_container_width=True):
    stop_webcam()

st.markdown("---")

# Streamlit UI for Video Upload Emotion Detection
st.markdown("### Upload Video for Emotion Detection")
st.markdown("Upload a video file to detect emotions.")

# Video file uploader
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"], label_visibility="collapsed")

# Trigger video upload and processing
if st.button("Upload and Detect Emotion") and uploaded_file:
    with st.spinner("Processing..."):
        upload_and_detect_emotion(uploaded_file)
