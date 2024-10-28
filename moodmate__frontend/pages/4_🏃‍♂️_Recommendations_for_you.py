import streamlit as st
import requests
from dotenv import load_dotenv
import os
from streamlit_extras.switch_page_button import switch_page
from utils.emotions import get_max_voted_emotion
load_dotenv()
API_BASE_URL = os.getenv("API__URL") 

# Streamlit configuration
st.set_page_config(page_title="Recommendations", page_icon="🏃‍♂️", initial_sidebar_state="expanded", layout='wide')

st.sidebar.markdown('# 😎 MoodMate')
st.sidebar.markdown('---')
st.sidebar.page_link('Home.py', label='Home')
st.sidebar.page_link('pages/1_📘_Type_your_thoughts.py', icon='🖋️')
st.sidebar.page_link('pages/2_🎙️_Speak_your_thoughts.py', icon='🎙️')
st.sidebar.page_link('pages/3_🙂_Express_your_thoughts.py', icon='🧑')
st.sidebar.page_link('pages/4_🏃‍♂️_Recommendations_for_you.py', icon='🏃‍♂️')
st.sidebar.markdown('---')

if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        switch_page('home')
        
def get_recommendation(emotion):
    
    
    try:
        # Prepare the data payload
        user_data = st.session_state['user']
        
        # Send POST request to the Flask API
        response = requests.post(f"{API_BASE_URL}/recommend", json=user_data, params={'emotion':emotion})
        
        if response.status_code == 200:
            return response.json() 
        else:
            st.error("Error in fetching recommendations. Please try again.")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit UI

st.title("🏃‍♂️ Personalized Activity Recommendation")
st.markdown("---")
c1, c2 = st.columns(2)

c1.image('https://cdn.prod.website-files.com/62693896276250215054cdda/63d3c0f75567c704a38fd62c_regular%20couple%20acitivty%20ritual%20cover%203.webp')


c2.markdown("### Get recommendations tailored for you!")
# Emotion input
emotion = c2.text_input("Your Emotion:", get_max_voted_emotion())

# Button to trigger recommendation retrieval
if c2.button("Get Recommendation", type='primary') and emotion:
    with st.spinner("Getting recommendation tailored for you ..."):
        recommendation_result = get_recommendation(emotion)

    if recommendation_result:
        recomendation = recommendation_result.get("recomendation")
        description = recommendation_result.get("description")
        
        if recomendation and description:
            c2.markdown(f"#### {recomendation}")
            c2.markdown(f"{description}")
        else:
            c2.warning("No recommendations found for the given emotion.")
            

