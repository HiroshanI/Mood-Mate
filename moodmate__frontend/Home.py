import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import requests 
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="MyRecommender", page_icon="😎", layout="wide", initial_sidebar_state="expanded")

st.sidebar.markdown('# 😎 MoodMate')
st.sidebar.markdown('---')
st.sidebar.page_link('Home.py', label='Home')
st.sidebar.page_link('pages/1_📘_Type_your_thoughts.py', icon='🖋️')
st.sidebar.page_link('pages/2_🎙️_Speak_your_thoughts.py', icon='🎙️')
st.sidebar.page_link('pages/3_🙂_Express_your_thoughts.py', icon='🧑')
st.sidebar.page_link('pages/4_🏃‍♂️_Recommendations_for_you.py', icon='🏃‍♂️')
st.sidebar.markdown('---')

API_BASE_URL = os.getenv("API__URL")

# Help
st.sidebar.subheader("Help")
st.sidebar.info("""
    - **Home**: Overview of the application.
    - **Enter Text**: Input your text for classification.
    - **Upload Audio**: Upload your audio files for analysis.
    - **Upload Video**: Analyze video content by uploading files.
    - **Get Recommendations**: Get suggestions based on your input.
""")

st.title("😎 Welcome to MoodMate")
st.markdown("""MoodMate offers personalized activity recommendations designed to uplift your mood, reduce stress, and enhance your overall well-being. 
            Whether you're feeling joyful, sad, or anything in between, MoodMate is here to help you navigate your emotions with tailored suggestions.""")
st.markdown("---")

if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
    st.markdown("##### Please :blue-background[🔓 Sign in] to continue ")
    st.markdown("##### Or if you don't have an account yet :red-background[🤝 Join us] ")
    c1, c2 = st.columns(2)
    if c1.button("Sign In", type='primary', use_container_width=True):
        switch_page('sign in')
    if c2.button("Register", use_container_width=True):
        switch_page('sign in')
    
else: 

    # Card template with better styles
    card_style = """
        <div style='
            height: {height}px;  
            max-height: {height}px; 
            border: 1px solid black;
            border-radius: 15px;
            position: relative;
            overflow: hidden;
            cursor:pointer; 
            text-decoration:none;
            margin-bottom:1rem;
            '>
            <div style="z-index:2;display:flex column nowrap; gap:1.5rem;position: absolute; left:1rem;right:1rem;bottom:0rem;padding:1.5rem;text-shadow:1px 1px 5px black">
                <h2 style='text-align: left; z-index: 1;'>{title}</h2>
                <h5 style='text-align: left; z-index: 1;'>{description}</h5>
            </div>
            <div style="z-index:1;background:{background_color} ;position:absolute; left:0;top:0;right:0;bottom:0;margin:0"></div>
            <img src='{background_image_url}' style='height:auto;width:auto;min-height:100%;min-width:100%';margin:auto;/>
        </div>
    """

    # Image URLs
    text_image_url = "https://betterlyf-upload.s3.amazonaws.com/1672910170319.jpg"
    audio_image_url = "https://imnet.com/wp-content/uploads/2022/02/Whatsapp-recorded-messages-_EN.jpg"
    video_image_url = "https://media.istockphoto.com/id/1314904275/photo/webcam-view-of-happy-smiling-businessman-teacher-talking-to-employees-students-looking-at.jpg?s=612x612&w=0&k=20&c=bAAtbxW-N4ITUNR0b_BjjQxBIyTP3PZn_cCywgMcDSc="
    recommend_image_url = "https://www.shutterstock.com/image-photo/collage-about-fit-men-women-260nw-2013386267.jpg"

    # First Row
    st.subheader("How are you feeling right now?")

    col1, col2, col3 = st.columns(3)

    card_data = [
        {"title": "Write your Thoughts", "action":'Start Writing',
            "description": "Write what you feel. Start a daily diary to reflect on your emotions.", 
            "height": 300, 'page':'Type Your Thoughts',
            "background_image_url": text_image_url, "background_color":"rgba(0, 37, 76, 0.6)"},
        {"title": "Voice your thoughts", "action":'Upload Audio',
            "description": "Upload a recording of your thoughts", 
            "height": 300, 'page':'Speak Your Thoughts',
            "background_image_url": audio_image_url, "background_color":"rgba(76, 44, 0, 0.6)"},
        {"title": "Show your thoughts", "action":'Live / Upload Video',
            "description": "Express your thoughts through video", 
            "height": 300, 'page':'Express Your Thoughts',
            "background_image_url": video_image_url, "background_color":"rgba(15, 76, 0, 0.6)"}
    ]

    for col, card in zip([col1, col2, col3], card_data):
        with col:
            st.markdown(card_style.format(**card), unsafe_allow_html=True)
            if st.button(card['action'], key=card['title'], type='primary', use_container_width=True):
                switch_page(card['page'])
                

    # Second Row
    st.markdown("---") 
    st.subheader("Activity Recommendations made for you")

    fourth_card = {
        "title": "Recommendations For You","action":'Find Recommendations',
        "description": "Discover personalized activities based on you and your mood.",
        "height": 300, 'page':'Recommendations for you',
        "background_image_url": recommend_image_url, "background_color":"rgba(76, 0, 0, 0.5)"
    }

    st.markdown(card_style.format(**fourth_card), unsafe_allow_html=True)
    if st.button(fourth_card['action'], key=fourth_card['title'], use_container_width=True, type='primary'):
        switch_page(fourth_card['page'])
        
# Footer
st.sidebar.markdown("---")
if st.sidebar.button("Logout", use_container_width=True):
    res = requests.get(API_BASE_URL+"/clear")
    st.session_state['authenticated'] = False
    st.session_state['user_data'] = None

st.sidebar.markdown("Made with ❤️ /RP030")
        