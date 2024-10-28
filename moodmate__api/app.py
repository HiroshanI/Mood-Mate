from flask import Flask, render_template, request, Response, session, redirect, url_for, jsonify
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user
from prediction.utils import get_max_voted_emotion
from datetime import datetime
# db
from pymongo import MongoClient
from bson import ObjectId
# text-clf
from prediction.text.evaluate import make_prediction
from prediction.text.augment import augment_sentences, classify_augmentations, extract_saliency
# image-clf
import cv2
from tensorflow.keras.models import model_from_json
from mtcnn import MTCNN
from prediction.video.camvideo import *
# audio-clf
from prediction.audio.audio_classification import classify_audio
# recommender
import keras
from openai import OpenAI
from prediction.recommender.utils import (predict_recommendations , 
                                          select_recomendation ,
                                          generate_random_data , 
                                          generate_query_from_json_excluding,
                                          create_gpt3_prompt, get_metadata, get_ratings) 

# Environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('APP__SECRET_KEY')
app.config['UPLOAD_FOLDER'] = 'uploads'

# Database configuration for MongoDB
mongo_client = MongoClient(f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}.x9xif.mongodb.net/?retryWrites=true&w=majority&appName={os.getenv('DB_CLUSTER_NAME')}")
db = mongo_client[os.getenv('DB_NAME')]
users_collection = db["users"]
print("Database connection established")

# User model for Flask login
class MongoUser(UserMixin):
    def __init__(self, user_data):
        self.user_data = user_data
        self.id = str(user_data['_id'])
        
    def __getattr__(self, name):
        return self.user_data.get(name)
    
    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            'age': self.age,
            'sex': self.sex,
            'location': self.location,
            'relationship_status': self.relationship_status,
            'designation': self.designation,
            'salary': self.salary,
            'likes': self.likes,
            'dislikes': self.dislikes,
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'notes':self.notes
        }
    
    @property
    def is_active(self):
        return True

# Login manager (Flask)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'signin'

@login_manager.user_loader
def load_user(user_id):
    user_data = db.users.find_one({"_id": ObjectId(user_id)})
    if user_data:
        return MongoUser(user_data)  # Return an instance of MongoUser
    return None

# @login_requir
# 
# ed
@app.route('/')
def home():
    return jsonify({"message":"Welcome to MoodMate!"})
    # return render_template('index.html', 
    #                        text_clf=session.get('text_clf_emotion'), 
    #                        image_clf=session.get('image_clf_emotion'),
    #                        audio_clf=session.get('audio_clf_emotion'),
    #                        common=get_max_voted_emotion([session.get('text_clf_emotion'),
    #                                                      session.get('image_clf_emotion'),
    #                                                      session.get('audio_clf_emotion')]))
    
# @login_required
@app.route('/user_info')
def user_info():
    return render_template('user_info.html', user=current_user)

# @login_required
@app.route('/clear')
def clear():
    session.clear()
    return jsonify({'message':'Logged out'})


# @login_required
@app.route('/text_classification', methods=['GET', 'POST'])
def text_classification():
    
    # Get session data for inputs (if available)
    model = session.get('text_clf_model')
    lang = session.get('lang')
    input_text = session.get('text_clf_input') if session.get('text_clf_input') else ""
    labels = ['🥺 Sadness','😃 Joy', '😍 Love', '😡 Anger','😱 Fear','😯 Surprise']
    labels_short = ['Sad', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
    sintam_labels = ['disgust','anger','fear','surprise','happy','sadness' ]
    sintam_labels_short = ['Disgust','Anger','Fear','Surprise','Joy','Sad' ]
    sentence1, sentence2 = session.get('s1'), session.get('s2')
    s1label, s2label = session.get('s1label'), session.get('s2label')
    predictions = session.get('text_predictions')
    augmented_sentences = session.get('augmented_sentences') if session.get('augmented_sentences') else []
    augmented_labels = session.get('augmented_labels') if session.get('augmented_labels') else []

    if request.method == 'POST':
        
        # TEXT CLASSIFICATION
        if 'input_text' in request.form:
            
            print("\nText Classfication Started")
            input_text = request.form['input_text']
            model = request.form['model_select'].split('_')[0]
            lang = request.form['model_select'].split('_')[1]
            session['text_clf_input'] = input_text
            session['text_clf_model'] = model
            session['lang'] = lang
            pred, confs= make_prediction(input_text, model=model, lang=lang)
            c = confs.tolist()[0]
            if lang == 'en':
                predictions = {
                    l:round(c[i]*100, 2) for i, l in enumerate(labels)
                }
                prediction = labels[pred]
                session['text_clf_emotion'] = labels_short[pred]
                session['text_predictions'] = predictions
                print('\nFunction: TEXT-CLF',
                  f'Model: {model}',
                  f'Input: {input_text}', 
                  f'Prediction: {prediction} @ {c[pred]*100:.2f}%\n', sep='\n')
                
                email = request.form.get('email')
                note_entry = {
                    "text": input_text,
                    "emotion": labels_short[pred],
                    "date_created": datetime.now()  # Use UTC for consistency
                }
                db.users.update_one(
                    {"email": email},
                    {"$push": {"notes": note_entry}}
                )
                user_data = db.users.find_one({"email": email})
                user = MongoUser(user_data)
                return {"pred":labels_short[pred],"confs": predictions, "user_data":user.to_dict()}
            else:
                predictions = {
                    l:round(c[i]*100, 2) for i, l in enumerate(sintam_labels)
                }
                prediction=sintam_labels[pred]
                session['text_clf_emotion'] = sintam_labels_short[pred]
                session['text_predictions'] = predictions
                print('\nFunction: TEXT-CLF',
                  f'Model: {model}',
                  f'Input: {input_text}', 
                  f'Prediction: {prediction} @ {c[pred]*100:.2f}%\n', sep='\n')
                email = request.form.get('email')
                note_entry = {
                    "text": input_text,
                    "emotion": sintam_labels_short[pred],
                    "date_created": datetime.now()  # Use UTC for consistency
                }
                db.users.update_one(
                    {"email": email},
                    {"$push": {"notes": note_entry}}
                )
                user_data = db.users.find_one({"email": email})
                user = MongoUser(user_data)
                return {"pred":sintam_labels_short[pred],"confs": predictions, "user_data":user.to_dict()}
            
                
            
            
        # DATA AUGMENTATION
        elif 'sentence1' in request.form and 'sentence2' in request.form:
            
            print("\nData Augmentation Started")
            
            # get input <- form
            sentence1 = request.form['sentence1']
            sentence2 = request.form['sentence2']
            s1label, s2label = request.form['sentence1_label'], request.form['sentence2_label']
            session['s1'] = sentence1
            session['s2'] = sentence2
            session['s1label'], session['s2label'] = s1label, s2label
            lang = request.form['shap_model']
            modname, tokname = "mod_en_89",'tok_en_89' # default
            t = int(request.form.get('thresh')) # threshold for salient token extraction
            
            # extract saliency
            p=0.6
            shaps,imp1, imp2 = extract_saliency(t, p, lang, sentence1, sentence2, s1label, s2label, labels, sintam_labels)
            print("Shaps", len(shaps))
            print(f"Model: {model} -> {modname}, {tokname} | Threshold: {t}" )
            
            # augment with SSwap
            factor=10
            augmented_sents = augment_sentences(factor, s1label, s2label, sentence1, sentence2, shaps, t, p)

            # classify augmented sentences
            augmented_sentences, augmented_labels = classify_augmentations(augmented_sents, labels, lang)
            session['augmented_sentences'] = augmented_sentences
            session['augmented_labels'] = augmented_labels
            print(f"Augmentations: {len(augmented_sentences)} contains {pd.Series(augmented_labels).unique()}")
            
            return jsonify({
                "augmented_sentences":zip(augmented_sentences, augmented_labels),
                "sentence1":[sentence1, s1label], "sentence2":[sentence2, s2label]
            })
                    
    # return render_template('text_classification.html', 
    #                        model_input=f"{model}_{lang}",
    #                        labels=labels, 
    #                        predictions=predictions, 
    #                        augmented_sentences=zip(augmented_sentences, augmented_labels),
    #                        input_text=input_text, sentence1=[sentence1, s1label], sentence2=[sentence2, s2label])



# @login_required
@app.route('/audio_classification', methods=['POST', 'GET'])
def audio_classification():
    classification_result = session.get('audio_clf_emotion')
    if request.method == 'POST':
        classification_result = ""
        file = request.files['audio_file']
        
        if file:
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            print(f'[AUDIO-CLF] Uploaded file saved: {filepath}')
            
            prediction = classify_audio(filepath)
            print("[AUDIO-CLF]: Prediction =", prediction)
            classification_result=prediction
            session['audio_clf_emotion'] = prediction
            return classification_result

    # return render_template("audio_classification.html", classification_result=classification_result)

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

detector = MTCNN()
background = np.random.rand(1, 48, 48, 3)
explainer = shap.DeepExplainer(emotion_model, background)

json_file = open('prediction/video/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("prediction/video/emotion_model.weights.h5")
print("[VIDEO-CLF] Load model : DONE")

cap = None
generalized_emotion = None
total_frames = 0
emotion_counts = {}
confidence_sums = {}

# @login_required
@app.route('/image_classification')
def image_classification():
    global generalized_emotion
    emotion = generalized_emotion  # Save the current emotion to a local variable
    return emotion

# Video streaming generator for webcam
def gen_frames():
    global cap, total_frames, emotion_counts, confidence_sums
    cap = cv2.VideoCapture(0)
    
    # Initialize emotion tracking
    emotion_counts = {emotion: 0 for emotion in emotion_dict.values()}
    confidence_sums = {emotion: 0 for emotion in emotion_dict.values()}
    prev_emotion = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        else:
            # Process the frame
            frame, prev_emotion = process_frame(frame, emotion_model, explainer, emotion_dict, total_frames, [], emotion_counts, confidence_sums, prev_emotion)
            total_frames += 1

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop', methods=['POST'])
def stop_webcam():
    global cap, generalized_emotion, emotion_counts, confidence_sums
    cap.release()
    generalized_emotion = calculate_generalized_emotion(emotion_counts, confidence_sums)
    return generalized_emotion
    # session['image_clf_emotion']= generalized_emotion
    # return redirect(url_for('image_classification'))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global generalized_emotion
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        # Process the video for emotion detection
        generalized_emotion = process_uploaded_video(filepath)
        return generalized_emotion

def process_uploaded_video(filepath):
    global emotion_counts, confidence_sums
    cap = cv2.VideoCapture(filepath)
    emotion_counts = {emotion: 0 for emotion in emotion_dict.values()}
    confidence_sums = {emotion: 0 for emotion in emotion_dict.values()}
    prev_emotion = None
    total_frames = 0

    frame_skip_rate = 25  # Process every 10th frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if total_frames % frame_skip_rate == 0:  # Process every nth frame
            frame, prev_emotion = process_frame(frame, emotion_model, explainer, emotion_dict, total_frames, [], emotion_counts, confidence_sums, prev_emotion)
        total_frames += 1

    cap.release()
    generalized_emotion = calculate_generalized_emotion(emotion_counts, confidence_sums)
    return generalized_emotion

# Recommendation routes starts here ------------------------------------
 
# Load model
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
model = keras.models.load_model('./prediction/recommender/dqn_model.keras')
print('[Recommender] Load model : DONE')

client = OpenAI(
    os.getenv('OPEN_API_KEY')
)

def chat_with_gpt(user_input= "hi Good Evening"):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "assistant", "content": user_input}]
    )
    content = response.choices[0].message.content
    print(f"[Recommender] Response: {response}")
    return str(content)
    

# @login_required
@app.route('/recommend/check' , methods=['POST'])
def load_and_pred():
    data = request.get_json()
    recomendation = data.get('rec') 
    random_data = generate_random_data()
    input_query = generate_query_from_json_excluding(random_data, recomendation)
    response = chat_with_gpt(input_query)
    return response

# @login_required
@app.route('/recommend', methods=['GET', 'POST'])
def get_recomendation():
    if request.method == 'POST':
        user_data = request.get_json()
        # print(request.args.get('emotion'))
        metadata = get_metadata()  
        ratings = get_ratings()
        recommended_action = predict_recommendations(model, metadata, ratings)
        recomendation = select_recomendation(recommended_action)
        user_data['_id'] = ObjectId(user_data['id'])
        user = MongoUser(user_data)
        random_data = generate_random_data(user)
        random_data['emotion'] = request.args.get('emotion')
        # input_query = generate_query_from_json_excluding(random_data, recomendation)
        # print('[Recommender] User data:', random_data.items())
        input_query = create_gpt3_prompt(random_data, recomendation)
        response = chat_with_gpt(input_query)
        
        return jsonify({
            "recomendation": recomendation,
            "description" : response
        })

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)  # Hash the password for security
        age = int(request.form['age'])
        sex = request.form['sex']
        location = request.form['location']
        relationship_status = request.form['relationship_status']
        designation = request.form['designation']
        salary = int(request.form['salary'])
        likes = request.form['likes']
        dislikes = request.form['dislikes']
        strengths = request.form['strengths']
        weaknesses = request.form['weaknesses']

        print('Likes', likes)

        # Prepare the new user data to insert into the MongoDB collection
        new_user = {
            'name': name,
            'email': email,
            'password': hashed_password,  # Save hashed password
            'age': age,
            'sex': sex,
            'location': location,
            'relationship_status': relationship_status,
            'designation': designation,
            'salary': salary,
            'likes': likes,
            'dislikes': dislikes,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'notes':[],
            'is_active': True  # Always allow the user to be active
        }

        try:
            # Check for existing user with the same email
            if db.users.find_one({'email': email}):
                print("[SIGNUP] Email already exists:", email)
                # return render_template('signup.html', error="Email already exists")
                return Response.status_code(400)

            # Insert the new user into the MongoDB collection
            result = db.users.insert_one(new_user)
            new_user['_id'] = result.inserted_id
            print("[SIGNUP] New user created:", name, email)

            # Convert to MongoUser and log the user in
            user = MongoUser(new_user)
            login_user(user)
            return jsonify({'user_data':user.to_dict()})

            # return render_template('index.html')  # Redirect or render success page

        except Exception as e:
            print("[SIGNUP] Error creating user:", e)
            return Response.status(400)
            # return render_template('register.html', error="An error occurred during signup")

    # return render_template('register.html')

@app.route('/signin', methods=['POST'])
def signin():
    email = request.form.get('email')
    password = request.form.get('password')
    
    # Retrieve user data from MongoDB
    user_data = db.users.find_one({"email": email})

    # Check if the user exists and verify the password
    if user_data and check_password_hash(user_data['password'], password):
        # Wrap user data in MongoUser class
        user = MongoUser(user_data)
        
        # Successful sign-in
        login_user(user)
        # Return user data as JSON response
        return jsonify({
            'message': 'Sign in successful!',
            'user_data':user.to_dict()
        }), 200
    else:
        # Return error response if sign-in fails
        return jsonify({'message': 'Invalid email or password.'}), 401

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=False)