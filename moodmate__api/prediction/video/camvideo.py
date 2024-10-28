import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from mtcnn import MTCNN
import shap

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the model from JSON file
json_file = open('prediction/video/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("prediction/video/emotion_model.weights.h5")
print("[VIDEO-CLF] Load model: DONE")

# Initialize the MTCNN detector
detector = MTCNN()

# Initialize SHAP explainer with your model
background = np.random.rand(1, 48, 48, 3)
explainer = shap.DeepExplainer(emotion_model, background)

# Function to limit SHAP explanation to every nth frame
def get_shap_explanations(cropped_img, explainer, emotion_dict, emotion_prediction):
    shap_values = explainer.shap_values(cropped_img)
    explanations = []
    pred_emotion_index = int(np.argmax(emotion_prediction))
    pred_emotion = emotion_dict[pred_emotion_index]
    
    explanation = f"Predicted Emotion: {pred_emotion}\n"
    
    # Provide SHAP values for each channel and pixel
    for i in range(len(shap_values)):
        mean_shap_value = np.mean(shap_values[i])
        explanation += f"  Mean SHAP Value: {mean_shap_value:.4f}\n"
        sorted_indices = np.argsort(shap_values[i].flatten())[::-1]
        top_indices = sorted_indices[:5]  # Show top 5 features
        for idx in top_indices:
            channel = idx % 3
            row = (idx // 3) // 48
            col = (idx // 3) % 48
            channel_name = ["Red", "Green", "Blue"][channel]
            shap_value = shap_values[i].flatten()[idx]
            explanation += f"  - Feature (Channel: {channel_name}, Pixel: ({row}, {col})): {shap_value:.4f}\n"
    
    explanations.append(explanation)
    return pred_emotion, explanations

# Function to process each frame, track emotions, calculate generalized emotion, and apply SHAP
def process_frame(frame, emotion_model, explainer, emotion_dict, count, shap_explanations, emotion_counts, confidence_sums, prev_emotion=None):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)
    
    # Skip processing if no faces are detected
    if not faces:
        return frame, prev_emotion
    
    for face in faces:
        x, y, width, height = face['box']
        x, y = max(x, 0), max(y, 0)
        # Extract the region of interest (ROI) in grayscale
        roi_gray = cv2.cvtColor(frame[y:y+height, x:x+width], cv2.COLOR_BGR2GRAY)
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
        cropped_img = np.expand_dims(cropped_img, axis=0)

        # Predict the emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        pred_emotion = emotion_dict[int(np.argmax(emotion_prediction))]
        confidence_score = emotion_prediction[0][int(np.argmax(emotion_prediction))]

        # Update the emotion counts and confidence sums
        emotion_counts[pred_emotion] += 1
        confidence_sums[pred_emotion] += confidence_score

        # Calculate SHAP values less frequently (e.g., every 30th frame) or when emotion changes
        #if count % 30 == 0 or pred_emotion != prev_emotion:
        #    pred_emotion, explanations = get_shap_explanations(cropped_img, explainer, emotion_dict, emotion_prediction)
        #    shap_explanations.append(explanations)  # Store SHAP explanations

        # Display the predicted emotion
        cv2.putText(frame, f"Predicted Emotion: {pred_emotion}", (x + 5, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    
    return frame, pred_emotion

# Function to calculate the generalized emotion
def calculate_generalized_emotion(emotion_counts, confidence_sums):
    weighted_scores = {}
    total_emotion_count = sum(emotion_counts.values())  # Total number of detected emotions

    for emotion in emotion_counts:
        if emotion_counts[emotion] > 0:
            # Calculate weighted score considering confidence and proportion of this emotion in total counts
            weighted_scores[emotion] = confidence_sums[emotion] * (emotion_counts[emotion] / total_emotion_count)

    # Determine the most generalized emotion
    generalized_emotion = max(weighted_scores, key=weighted_scores.get)
    return generalized_emotion
