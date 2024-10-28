import streamlit as st
import requests
from dotenv import load_dotenv
import os
from streamlit_extras.switch_page_button import switch_page 

# Load environment variables
load_dotenv()

# Retrieve the API URL from the environment variables
FLASK_API_URL = os.getenv("API__URL")

# Set page configuration
st.set_page_config(page_title="MyRecommender", page_icon="😎", layout="centered")

# Create tabs for Sign In and Sign Up
tab1, tab2 = st.tabs(["🔓 Sign In", "🤝 Sign Up"])

# ------------------- Sign In Tab -------------------
with tab1:
    st.title("🔓 Sign In")
    st.subheader("Join us to reflect and improve your Emotional health")

    # Create a form for Sign In
    with st.form(key='signin_form'):
        email = st.text_input("Email", placeholder="Enter your email")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submit_button = st.form_submit_button("Sign In")

    if submit_button:
        # Make a POST request to the Sign In endpoint
        response = requests.post(f"{FLASK_API_URL}/signin", data={'email': email, 'password': password})
        if response.status_code == 200:
            st.success("Sign in successful!")
            user = response.json()  # Assuming your Flask API returns user data in JSON format
            st.session_state['authenticated'] = True
            st.session_state['user'] = user['user_data']  # Store user data in session state
            switch_page("home")  # Redirect to the home page
        else:
            st.error("Invalid email or password.")

# ------------------- Sign Up Tab -------------------
with tab2:
    st.title("🤝 Sign Up")
    st.subheader("Create an account to get started")

    # Create a form for Sign Up
    with st.form(key='signup_form'):
        name = st.text_input("Name", placeholder="Enter your full name")
        email = st.text_input("Email", placeholder="Enter your email")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        age = st.number_input("Age", min_value=1, step=1, format="%d")
        sex = st.selectbox("Sex", ["Male", "Female", "Other"])
        location = st.text_input("Location", placeholder="Enter your location")
        relationship_status = st.selectbox("Relationship Status", ["Single", "In a relationship", "Married", "Other"])
        designation = st.text_input("Designation", placeholder="Enter your job title")
        salary = st.number_input("Salary", min_value=0, step=1, format="%d")
        likes = st.text_area("Likes", placeholder="Enter your likes")
        dislikes = st.text_area("Dislikes", placeholder="Enter your dislikes")
        strengths = st.text_area("Strengths", placeholder="Enter your strengths")
        weaknesses = st.text_area("Weaknesses", placeholder="Enter your weaknesses")
        
        # Submit button for the form
        signup_button = st.form_submit_button("Sign Up")
    
    # If the form is submitted
    if signup_button:
        # Prepare the form data
        signup_data = {
            "name": name,
            "email": email,
            "password": password,
            "age": age,
            "sex": sex,
            "location": location,
            "relationship_status": relationship_status,
            "designation": designation,
            "salary": salary,
            "likes": likes,
            "dislikes": dislikes,
            "strengths": strengths,
            "weaknesses": weaknesses
        }
        
        # Make a POST request to the Sign Up endpoint
        try:
            response = requests.post(f"{FLASK_API_URL}/signup", data=signup_data)
            
            # Check if the signup was successful
            if response.status_code == 200:
                st.success("Signup successful! You are now logged in.")
                user_data = response.json()  # Assuming your API returns user data
                st.session_state['authenticated'] = True
                st.session_state['user'] = user_data
                switch_page("home")  
            elif response.status_code == 400:
                st.error("Email already exists. Please use a different email.")
            else:
                st.error("An error occurred during signup. Please try again.")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
