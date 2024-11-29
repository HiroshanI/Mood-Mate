import streamlit as st
import requests
from dotenv import load_dotenv
import os
import pandas as pd
import plotly.graph_objects as go

from utils.emotions import standardize_emotion_label, get_encouraging_sentence

load_dotenv()

API_BASE_URL = os.getenv("API__URL")

# Set page configuration
st.set_page_config(page_title="Type Your Thoughts", page_icon="💬", layout='wide', initial_sidebar_state="expanded")


st.sidebar.markdown('# 😎 MoodMate')
st.sidebar.markdown('---')
st.sidebar.page_link('Home.py', label='Home')
st.sidebar.page_link('pages/1_📘_Type_your_thoughts.py', icon='🖋️')
st.sidebar.page_link('pages/2_🎙️_Speak_your_thoughts.py', icon='🎙️')
st.sidebar.page_link('pages/3_🙂_Express_your_thoughts.py', icon='🧑')
st.sidebar.page_link('pages/4_🏃‍♂️_Recommendations_for_you.py', icon='🏃‍♂️')
st.sidebar.markdown('---')

# Title
st.markdown("# 💬 Type Your Thoughts")
st.markdown("---")

c1, c2 = st.columns(2)

c2.image("https://st4.depositphotos.com/20596548/39820/i/450/depositphotos_398207466-stock-photo-young-man-writes-notebook-table.jpg")

# Text Classification Section
c1.markdown("### Hi, how are you doing today?")
input_text = c1.text_area("", 
                           height=200, 
                           placeholder="Type or paste your text here...")

# Model selection
model_options = [
    "augmented_en", 
    "augmented_si", 
    "augmented_tm",
    "original_en", 
    "original_si", 
    "original_tm", 
]

st.sidebar.markdown("## ⚙️ Extra Configurations")
model_select = st.sidebar.selectbox("Select a different language / model", options=model_options)

# Toggle button for Data Augmentation
show_data_augmentation = st.sidebar.checkbox("Show Data Augmentation", value=False)

predictions = None

def get_emotion(input_text, model_select):
    try:
        # Prepare the data payload
        data = {"input_text": input_text, 
                "model_select": model_select,
                'email':st.session_state['user']['email']}
        
        # Send POST request to the Flask API
        response = requests.post(f"{API_BASE_URL}/text_classification", data=data)
        
        # Handle the response
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Error in classifying yout text. Please try again later.")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

if c1.button("Add to Diary", type="primary", use_container_width=True):
    response = get_emotion(input_text, model_select)
    if response:
        emotion = standardize_emotion_label(response['pred'])
        s = get_encouraging_sentence(emotion)
        st.info(s)
        st.page_link('pages/4_🏃‍♂️_Recommendations_for_you.py', icon='👉')
        st.session_state['text_last_clf'] = emotion
        predictions = response['confs']
        st.session_state['user'] = response['user_data']
        predictions = {k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)}

# Expander for Classification Results
with c1.expander("Classification Results", expanded=False):
    # Display prediction results
    if predictions:
        max_confidence = max(predictions.values())
        for label, confidence in predictions.items():
            st.write(f"{label}: {confidence}%")
            st.progress(confidence / 100)  # Progress bar for confidence
    else:
        for l in ["🥺 Sadness", "😃 Joy", "😍 Love", "😡 Anger", "😱 Fear", "😯 Surprise"]:
            st.write(f"{l}: 0%")
            st.progress(0)
            
# Charting Section
st.subheader("My Emotions Chart")

user_notes = st.session_state['user']['notes']
if user_notes is not None:
    notes_data = user_notes
    
    if notes_data:
        # Convert notes to a DataFrame for easy plotting
        df = pd.DataFrame(notes_data)
        df['emotion'] = df['emotion'].apply(standardize_emotion_label)
        df['date_created'] = pd.to_datetime(df['date_created'])
        df = df.sort_values(by='date_created')
        
        emotion_mapping = {
            'Anger': 'Anger 😡',
            'Fear': 'Fear 😱',
            'Sadness': 'Sadness 🥺',
            'Joy': 'Joy 😃',
            'Love': 'Love 😍',
            'Surprise': 'Surprise 😯',
            'Disgust': 'Disgust 🤮'
        }
        df['emotion_label'] = df['emotion'].map(emotion_mapping)
        print(df)
        # Create a Plotly figure
        fig = go.Figure()
        
        # Add a line plot for emotions over time
        fig.add_trace(go.Scatter(
            x=df['date_created'],
            y=df['emotion_label'],
            mode='lines+markers',
            line=dict(width=4, color='violet'),
            marker=dict(size=20, color='violet'),
            name='Emotions',
            text=df['emotion'],
            hoverinfo='text'
        ))
        
        fig.update_yaxes(categoryorder='array', categoryarray=list(emotion_mapping.values()),
                         tickfont=dict(size=20))
        
        # Update layout for dark theme
        fig.update_layout(
            # title='Emotions Over Time',
            # title_font=dict(size=16, color='white'),
            xaxis_title='Date',
            xaxis_title_font=dict(size=12, color='white'),
            yaxis_title='Emotion',
            yaxis_title_font=dict(size=12, color='white'),
            paper_bgcolor='#222222',
            plot_bgcolor='#222222',
            font=dict(color='white'),
            xaxis=dict(showgrid=True, gridcolor='dimgrey'),
            yaxis=dict(showgrid=True, gridcolor='dimgrey')
        )

        # Show plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No journals found. Start writing your first journal 📔")

# Display Notes Below the Graph
st.subheader("My Diary")

if user_notes is not None:
    notes_data = user_notes
    
    if notes_data:
        # Sort notes by date created (optional: latest first)
        notes_data = sorted(notes_data, key=lambda x: x['date_created'], reverse=True)
        
        for note in notes_data:
            # Extract details
            note_text = note.get('text', 'No text')
            date_created = pd.to_datetime(note.get('date_created')).strftime('%Y-%m-%d %H:%M')
            emotion = standardize_emotion_label(note.get('emotion'))
            
            # Display each note in an expander
            with st.expander(f"{date_created}", expanded=False):
                st.markdown(f"##### {note_text}")
                st.markdown(f"**Emotion Detected**: {emotion_mapping[emotion]}")
                
    else:
        st.info("No journal entries found. Start writing your first journal 📔")



# Data Augmentation Section
if show_data_augmentation:
    st.header("Data Augmentation")
    col1, col2 = st.columns(2)

    with col1:
        # Sentence 1
        sentence1 = st.text_area("Sentence 1", placeholder="Enter the first sentence", height=100)
        sentence1_label = st.selectbox("Select Label for Sentence 1", 
                                        options=['🥺 Sadness', '😃 Joy', '😍 Love', '😡 Anger', '😱 Fear', '😯 Surprise'])

        # Sentence 2
        sentence2 = st.text_area("Sentence 2", placeholder="Enter the second sentence", height=100)
        sentence2_label = st.selectbox("Select Label for Sentence 2", 
                                        options=['🥺 Sadness', '😃 Joy', '😍 Love', '😡 Anger', '😱 Fear', '😯 Surprise'])

        # Select model for augmentation
        shap_model = st.selectbox("Select Model", options=["en", "si", "tm"])

        # Threshold input
        threshold = st.number_input("Threshold", value=85)

        if st.button("Augment", type="primary", use_container_width=True):
            # Handle data augmentation logic
            st.success("Data augmentation logic executed!")

    with col2:
        st.subheader("Augmented Sentences")
        # Sample augmented sentences for display (replace with actual results)
        augmented_sentences = [
            ("Augmented Sentence 1", "🥺 Sadness"),
            ("Augmented Sentence 2", "😃 Joy"),
            ("Augmented Sentence 3", "😍 Love"),
        ]

        if augmented_sentences:
            for s, l in augmented_sentences:
                if l == sentence1_label:
                    st.success(s)
                else:
                    st.error(s)
