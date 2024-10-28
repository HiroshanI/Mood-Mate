import streamlit as st

def get_encouraging_sentence(emotion):
    """
    Returns an encouraging sentence based on the detected emotion.
    
    Parameters:
    - emotion (str): The standardized emotion label.
    
    Returns:
    - str: An encouraging sentence.
    """
    emotion_sentences = {
        "Sadness": "We sense that you're feeling sad. Let us suggest activities to help lift your mood. Give them a try, and see if they bring a little sunshine to your day.",
        "Anger": "It looks like you're experiencing anger. Take a moment to breathe, and let us recommend activities that might help you find calm and peace.",
        "Fear": "We noticed you're feeling afraid. Let us guide you with activities designed to bring comfort and ease to your mind.",
        "Love": "You're feeling love! That's wonderful. Let us suggest activities that can amplify this positive energy and keep the good vibes flowing.",
        "Joy": "It’s great to see you’re joyful! How about some activities to keep that joy shining even brighter?",
        "Surprise": "You seem surprised! Explore our recommendations to see how you can turn that surprise into something even more exciting.",
        "Disgust": "We sense some feelings of disgust. Let us offer activities to help shift your focus and bring positivity to your day.",
        "Neutral": "You're feeling neutral at the moment. Let's find activities that can add a touch of joy and inspiration to your day."
    }
    
    # Get the corresponding sentence, return a default if not found
    return emotion_sentences.get(emotion, "How are you feeling? Let us recommend activities that suit your current mood.")


def standardize_emotion_label(emotion):
    # Define the mapping dictionary
    emotion_mapping = {
        'sad': 'Sadness',
        'sadness': 'Sadness',
        'Sad': 'Sadness',
        'Sadness': 'Sadness',
        '🥲 Sad': 'Sadness',
        'anger': 'Anger',
        'angry': 'Anger',
        '😡 Anger': 'Anger',
        'mad': 'Anger',
        'rage': 'Anger',
        'fear': 'Fear',
        'afraid': 'Fear',
        'scared': 'Fear',
        '😱 Fear': 'Fear',
        'love': 'Love',
        '😍 Love': 'Love',
        'affection': 'Love',
        'joy': 'Joy',
        'happy': 'Joy',
        '😃 Joy': 'Joy',
        'happiness': 'Joy',
        'surprise': 'Surprise',
        'shocked': 'Surprise',
        '😯 Surprise': 'Surprise',
        'disgust': 'Disgust',
        'disgusted': 'Disgust',
        '🤢 Disgust': 'Disgust',
        'neutral': 'Neutral',
        'calm': 'Neutral',
        '😐 Neutral': 'Neutral'
    }
    
    # Convert the input emotion to lowercase and find the standardized label
    standardized_emotion = emotion_mapping.get(emotion.lower(), emotion)
    return standardized_emotion

def get_max_voted_emotion():
    # Fetch the last classification results from each input type
    video_emotion = st.session_state.get('video_last_clf')
    text_emotion = st.session_state.get('text_last_clf')
    audio_emotion = st.session_state.get('audio_last_clf')

    # Combine all available emotions into a list
    emotions = []
    if video_emotion:
        emotions.append(video_emotion)
    if text_emotion:
        emotions.append(text_emotion)
    if audio_emotion:
        emotions.append(audio_emotion)

    # If no emotions were classified, return None or a default message
    if not emotions:
        return "No emotions detected."

    # Count the occurrences of each emotion
    from collections import Counter
    emotion_counts = Counter(emotions)

    # Get the most common emotion
    max_voted_emotion = emotion_counts.most_common(1)[0][0]

    return max_voted_emotion