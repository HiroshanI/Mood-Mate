import os
import pandas as pd
import numpy as np

from pydub import AudioSegment, effects
import librosa
import noisereduce as nr

import matplotlib.pyplot as plt
from librosa import display
import IPython.display as ipd

from pydub.effects import normalize
from scipy.io import wavfile
import noisereduce as nr

import warnings
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model

# Function to extract MFCC features
# Capture the spectral characteristics of the audio signal
def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc_result = librosa.feature.mfcc(y=data, sr=sr)
    return np.ravel(mfcc_result) if flatten else mfcc_result

# Function to calculate Spectral Centroid
# Indicates the "center of mass" of the spectrum, providing information about the brightness of the sound
def spectral_centroid(data, sr, frame_length=2048, hop_length=512):
    spectral_centroid_result = librosa.feature.spectral_centroid(y=data, sr=sr)
    return np.ravel(spectral_centroid_result)

# Function to calculate Spectral Contrast
# Captures the difference in amplitude between peaks and valleys in the spectrum.
def spectral_contrast(data, sr, frame_length=2048, hop_length=512):
    spectral_contrast_result = librosa.feature.spectral_contrast(y=data, sr=sr)
    return np.ravel(spectral_contrast_result)

# Function to extract various audio features
def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = []

    # MFCC feature extraction
    mfcc_features = mfcc(data, sr, frame_length, hop_length)
    if mfcc_features.size > 0:
        result.append(mfcc_features)

    # Spectral Centroid feature extraction
    spectral_centroid_features = spectral_centroid(data, sr, frame_length, hop_length)
    if spectral_centroid_features.size > 0:
        result.append(spectral_centroid_features)

    # Spectral Contrast feature extraction
    spectral_contrast_features = spectral_contrast(data, sr, frame_length, hop_length)
    if spectral_contrast_features.size > 0:
        result.append(spectral_contrast_features)

    # Concatenate feature vectors
    if len(result) > 0:
        return np.hstack(result)
    else:
        return np.array([])



# Function to pitch-shift the audio
def pitch(data, sr, n_steps=2.0):
    pitched_audio = librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)
    return pitched_audio

# Function to add additive white Gaussian noise to the audio (AWGN)
def noise(data, noise_level=0.005):
    noise = np.random.normal(0, noise_level, len(data))
    return data + noise

# Function to get features from the audio file
def get_features(path, duration=2.5, offset=0.6, noise_level=0.005):
    data, sr = librosa.load(path, duration=duration, offset=offset)
    aud = extract_features(data)
    audio = np.array(aud)

    # Original audio
    noised_audio = noise(data, noise_level=noise_level)
    aud2 = extract_features(noised_audio)
    audio = np.vstack((audio, aud2))  # Add features of noised audio to the result

    # Pitch-shifted audio
    pitched_audio = pitch(data, sr)
    aud3 = extract_features(pitched_audio)
    audio = np.vstack((audio, aud3))  # Add features of pitch-shifted audio to the result

    # Pitch-shifted and noised audio
    pitched_audio1 = pitch(data, sr)
    pitched_noised_audio = noise(pitched_audio1, noise_level=noise_level)
    aud4 = extract_features(pitched_noised_audio)
    audio = np.vstack((audio, aud4))  # Add features of pitch-shifted and noised audio to the result

    return audio

def classify_audio(path):
    audio_ = get_features(path)
    X_ = audio_
    X_n = np.mean(X_, axis=0)
    X__n = (np.expand_dims(X_n, axis=0))  # Single prediction
    X_test_cnn = np.expand_dims(X__n, axis = 2)
    
    # Load your trained model
    model = load_model('prediction/audio/audio_cnn_lstm_model_n.h5')
    print("[AUDIO-CLF] Load model: DONE")
    
    prediction = model.predict(X_test_cnn)
    average_prediction = np.mean(prediction, axis=0)
    emotion = {
        '03' : 'happy',
        '01' : 'neutral',
        '02' : 'calm',
        '04' : 'sad',
        '05' : 'angry',
        '06' : 'fearful',
        '07' : 'sad',
        '08' : 'surprised'
    }
    predicted_class = np.argmax(average_prediction, axis = 0) + 1
    prediction_str = f'{predicted_class:02}'
    predicted_emotion = emotion.get(prediction_str, 'Unknown emotion')
    return predicted_emotion