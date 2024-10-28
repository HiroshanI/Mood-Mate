from collections import Counter

def get_max_voted_emotion(emotions):
    valid_emotions = [emotion for emotion in emotions if emotion is not None]
    if not valid_emotions:
        return None
    emotion_counts = Counter(valid_emotions)
    emotion = emotion_counts.most_common(1)[0][0]
    emot = ''
    if 'sad' in emotion.strip().lower():
        emot = 'Sadness'
    elif ('joy' in emotion.strip().lower()) or ('happy' in emotion.strip().lower()):
        emot = 'Joy'
    elif ('anger' in emotion.strip().lower()) or ('angry' in emotion.strip().lower()):
        emot = 'Anger'
    return emot