import numpy as np
import tensorflow as tf
import random

def predict_recommendations(model, metadata_input, ratings_input):

    try:
     
        metadata_tensor = tf.convert_to_tensor(metadata_input, dtype=tf.string)
        metadata_tensor = tf.reshape(metadata_tensor, (1, 1))  

        ratings_tensor = tf.convert_to_tensor(ratings_input, dtype=tf.float32)  

        # Make predictions using the model
        q_values = model.predict([metadata_tensor, ratings_tensor])
        #print("Q-values:", q_values)

        # Get recommended actions (top 10 actions based on Q-values)
        #recommended_actions = np.argpartition(q_values[0], -10)[-10:]  # Get top 10 actions
        recommended_action = np.argmax(q_values[0])
        #print("Recommended Actions:", recommended_actions)

        return recommended_action
    
    

    except Exception as e:
        print("Error during prediction:", e)
        return None, None
    

def select_recomendation(index):
    recommendations = [
    "Engage in a hobby you love.",
    "Practice deep breathing exercises.",
    "Write down your feelings.",
    "Take short breaks regularly.",
    "Call a friend or family member."
    ]
    
    return random.choice(recommendations)

def get_metadata():
    return random.choice([["1 0"], ["0 0"], ["0 1"], ["1 1"]])

def get_ratings():
    return np.array([[[random.choice([0,1,2,3,4,5,6,7,8,9])]]])

def generate_random_data(u):
    print(u)
    data = {
        'age': u.age,
        'sex': u.sex,
        'location': u.location,
        'relationship_status': u.relationship_status,
        'designation': u.designation,
        'salary': u.salary,
        'likes': [i.strip().lower() for i in u.likes.split(",")],
        'dislikes': [i.strip().lower() for i in u.dislikes.split(",")],
        'strengths': [i.strip().lower() for i in u.strengths.split(",")],
        'weaknesses': [i.strip().lower() for i in u.weaknesses.split(",")],
        # 'negative state of emotion': random.choice(['Frustration', 'Anger', 'Anxiety', 'Sadness']),
        # 'reason': random.choice(['Long work hours', 'Workplace conflicts', 'Lack of recognition', 'Monotony']),
        # 'suggestion': random.choice(['Take a walk in nature', 'Practice mindfulness', 'Talk to a friend', 'Listen to music'])
    }
    return data


def generate_query_from_json_excluding(json_data, exclude_key):
    # Extract the values we need for the summary, excluding the specified key
    filtered_data = {k: v for k, v in json_data.items() if k != exclude_key}
    
    # Create the query with the selected data
    query = (
        f"Generate a short recomendation about {exclude_key} for a person with the below attributes: "
        f"Age: {filtered_data['age']}, Sex: {filtered_data['sex']}, Location: {filtered_data['location']}, "
        f"Relationship Status: {filtered_data['Relationship Status']}, what advice would you suggest?"
    )
    
    return query

def create_gpt3_prompt(user_data, recomendation):
    # Format lists as comma-separated strings
    print(f'[Reccomender] User Data: {user_data.items()}')
    likes = ', '.join(user_data['likes'])
    dislikes = ', '.join(user_data['dislikes'])
    strengths = ', '.join(user_data['strengths'])
    weaknesses = ', '.join(user_data['weaknesses'])
    emotion = user_data['emotion'].capitalize()  # Capitalize for better readability
    
    prompt = f"""
**User Profile:**
- **Age:** {user_data['age']}
- **Sex:** {user_data['sex']}
- **Location:** {user_data['location']}
- **Relationship Status:** {user_data['relationship_status']}
- **Designation:** {user_data['designation']}
- **Salary:** Rs.{user_data['salary']:,}
- **Emotion:** {emotion}
- **Likes:** [{likes}]
- **Dislikes:** [{dislikes}]
- **Strengths:** [{strengths}]
- **Weaknesses:** [{weaknesses}]

- **emotion:** {emotion}
- **activity recommendation:** {recomendation}

**Task:**
Based on the above user profile, including the current emotional state and activity recommendation. Generate a short description explaining why this activity recommendation is suitable for the particular user based on the user's attributes and how it would alleivate the negative emotion of the user. Improve on the description based on the user's likes and strengths and how it can be incooperated with the recommendation. Keep the description's maximum tokens to 256. You can also include specific information related to location, occupation, age, salary for examples on the recommended activity only if it is appropriate. Ensure the examples are appropriate for the user's age.

**Output Format:**
description
"""
    return prompt

