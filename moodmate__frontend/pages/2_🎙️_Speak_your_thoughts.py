import streamlit as st
import requests
from dotenv import load_dotenv
import os

from utils.emotions import get_encouraging_sentence, standardize_emotion_label


load_dotenv()
API_BASE_URL = os.getenv("API__URL") 
# Streamlit configuration
st.set_page_config(page_title="Audio Classification", page_icon="🎧", initial_sidebar_state="expanded")


st.sidebar.markdown('# 😎 MoodMate')
st.sidebar.markdown('---')
st.sidebar.page_link('Home.py', label='Home')
st.sidebar.page_link('pages/1_📘_Type_your_thoughts.py', icon='🖋️')
st.sidebar.page_link('pages/2_🎙️_Speak_your_thoughts.py', icon='🎙️')
st.sidebar.page_link('pages/3_🙂_Express_your_thoughts.py', icon='🧑')
st.sidebar.page_link('pages/4_🏃‍♂️_Recommendations_for_you.py', icon='🏃‍♂️')
st.sidebar.markdown('---')

def upload_audio_file(audio_file):
    """Uploads the audio file to the Flask API and returns the classification result."""
    try:
        # Ensure the temporary upload folder exists
        upload_folder = "uploads"  # Same as your Flask app's upload folder
        os.makedirs(upload_folder, exist_ok=True)

        # Save the uploaded file to the temporary folder
        file_path = os.path.join(upload_folder, audio_file.name)
        with open(file_path, "wb") as f:
            f.write(audio_file.getbuffer())
        
        # Send the file to the Flask backend for classification
        with open(file_path, "rb") as f:
            files = {"audio_file": (audio_file.name, f, "audio/wav")}
            response = requests.post(f"{API_BASE_URL}/audio_classification", files=files)
        
        # Handle the response
        if response.status_code == 200:
            return response.text  # Get the classification result from the response
        else:
            st.error("Error in classification. Please try again.")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit UI
st.title("🎧 Speak your Thoughts")
st.markdown("Upload an audio file to detect emotions.")

# Audio file uploader
audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"], label_visibility="collapsed")

# Button to trigger audio classification
if st.button("Classify Audio") and audio_file:
    with st.spinner("Classifying..."):
        classification_result = upload_audio_file(audio_file)

    if classification_result:
        classification_result = standardize_emotion_label(classification_result)
        s = get_encouraging_sentence(classification_result)
        st.info(s)
        st.page_link('pages/4_🏃‍♂️_Recommendations_for_you.py', icon='👉')
        # st.success(f"{classification_result}")
        st.session_state['audio_last_clf'] = classification_result

