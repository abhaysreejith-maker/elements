from flask import Flask, render_template, redirect, request, url_for, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import os
import random
import base64
# Import libraries
try:
    import pyaudio
except ImportError:
    pyaudio = None

import librosa
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Mocking the model because tensorflow is not available for Python 3.14 yet
class MockModel:
    def predict(self, features):
        # Return random probabilities for 8 emotions
        res = np.random.rand(1, 8)
        res = res / res.sum()
        return res

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'static/audio'

@app.context_processor
def inject_user():
    return dict(logged_in_user=session.get('user_email'))

# SQLite connection
def get_db_connection():
    conn = sqlite3.connect('music_rec.db')
    return conn

def executionquery(query,values):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query,values)
    conn.commit()
    conn.close()
    return

def retrivequery1(query,values):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query,values)
    data = cursor.fetchall()
    conn.close()
    return data

def retrivequery2(query):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    return data


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/About')
def about():
    return render_template('about.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['useremail']
        password = request.form['password']
        c_password = request.form['c_password']
        username = request.form['username']
        age = request.form['age']
        gender = request.form['gender']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                hashed_password = generate_password_hash(password)
                query = "INSERT INTO users (email, password, username, age, gender) VALUES (?, ?, ?, ?, ?)"
                values = (email, hashed_password, username, age, gender)
                executionquery(query, values)
                return render_template('login.html', message="Successfully registered. Please sign in.")
            return render_template('register.html', message="This email is already registered.")
        return render_template('register.html', message="Confirm password does not match.")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['useremail']
        password = request.form['password']
        
        query = "SELECT password FROM users WHERE UPPER(email) = ?"
        values = (email.upper(),)
        password_data = retrivequery1(query, values)

        if password_data and check_password_hash(password_data[0][0], password):
            session['user_email'] = email
            return redirect(url_for('home'))
        return render_template('login.html', message="Invalid email or password.")
    return render_template('login.html')


@app.route('/home')
def home():
    if not session.get('user_email'):
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/demo_login')
def demo_login():
    session['user_email'] = 'demo@example.com'
    return redirect(url_for('home'))

# Load recommendation data
try:
    data = pd.read_csv('data_moods.csv')
    features = ['danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'valence', 'loudness', 'speechiness', 'tempo']
    feature_matrix = data[features]

    # Normalize the feature matrix
    scaler = StandardScaler()
    normalized_matrix = scaler.fit_transform(feature_matrix)

    # Apply SVD
    svd = TruncatedSVD(n_components=2)
    decomposed_matrix = svd.fit_transform(normalized_matrix)
    svd_matrix = np.dot(decomposed_matrix, svd.components_)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(svd_matrix)
except Exception as e:
    print(f"Error loading CSV or processing features: {e}")
    data = pd.DataFrame()
    similarity_matrix = np.array([])

def recommend_songs(predicted_mood, num_recommendations=10):
    mood_recommendations = {
        'happy': ['happy', 'energetic'],
        'energetic': ['energetic', 'happy'],
        'neutral': ['happy', 'calm'],
        'calm': ['calm', 'happy'],
        'sad': ['calm', 'happy'],
        'angry': ['calm', 'energetic'],
        'fearful': ['calm', 'happy'],
        'disguised': ['calm', 'happy'],
        'surprised': ['calm', 'happy']
    }

    recommended_moods = mood_recommendations.get(predicted_mood.lower(), ['happy', 'calm'])
    recommended_songs = []

    if similarity_matrix.size == 0 or data.empty:
        return []

    # Simple recommendation logic for demo
    filtered_songs = data[data['mood'].str.lower().isin(recommended_moods)]
    if not filtered_songs.empty:
        sample_size = min(len(filtered_songs), num_recommendations)
        songs = filtered_songs.sample(sample_size)
        for _, song in songs.iterrows():
            recommended_songs.append({
                'name': song['name'],
                'artist': song['artist']
            })

    return recommended_songs


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('user_email'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'audio' not in request.files:
            return render_template("upload.html", message="No file selected.")

        myfile = request.files['audio']
        if myfile.filename == '':
            return render_template("upload.html", message="No file selected.")

        accepted_formats = ['mp3', 'wav', 'ogg', 'flac']
        if not myfile.filename.split('.')[-1].lower() in accepted_formats:
            message = "Invalid file format. Accepted formats: {}".format(', '.join(accepted_formats))
            return render_template("upload.html", message=message)

        filename = secure_filename(myfile.filename)
        mypath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        myfile.save(mypath)

        result = record_audio(record=False, file_loc=mypath)
        mood = result
        recommendations = recommend_songs(predicted_mood=mood)
        return render_template('upload.html', prediction=result, path=mypath, recommendations=recommendations)

    return render_template('upload.html')



CHUNK = 1024*4
FORMAT = 8 # Represents pyaudio.paInt16
CHANNELS = 1
RATE = 48000

# Mocking model loading
model = MockModel()

#Extract features
def extract_features(file_name):
    try:
        X, sample_rate = librosa.load(file_name)
        stft = np.abs(librosa.stft(X))
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        contrast=np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        tonnetz=np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T, axis=0)
        return mfccs, chroma, mel, contrast, tonnetz
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return np.zeros(40), np.zeros(12), np.zeros(128), np.zeros(7), np.zeros(6)

#generating predictions
def speech_to_emotion(filename):
    mfccs, chroma, mel, contrast, tonnetz= extract_features(filename)

    f=np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    features=f.reshape(1, -1)

    his=model.predict(features)

    emotions=['neutral','calm','happy','sad','angry','fearful','disgused','surprised']
    y_pred=np.argmax(his, axis=1)
    pred_prob=np.max(his,axis=1)
    pred_emo=(emotions[y_pred[0]],pred_prob[0])
    return pred_emo

def record_audio(record=True, file_loc=None):
    if record:
        if pyaudio is None:
            return "Microphone recording not available (pyaudio missing)"
        # Real recording logic omitted for brevity in mock setup
        return "Recording finished (Demo)"
    else:
        if file_loc:
            emo=speech_to_emotion(file_loc)[0]
            return emo.capitalize()


from deepface import DeepFace
import cv2
# ... existing code ...
def predict_camera_emotion(image_bytes):
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Analyze the image
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        
        # The result is a list of dictionaries, one for each detected face.
        # We'll take the first one.
        if result and isinstance(result, list):
            dominant_emotion = result[0]['dominant_emotion']
            return dominant_emotion
        else:
            return "neutral"

    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return "neutral"

@app.route('/capture_mood', methods=['POST'])
def capture_mood():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    
    predicted_mood = predict_camera_emotion(image_bytes)
    
    session['predicted_mood'] = predicted_mood
    
    return jsonify({'redirect': url_for('recommendations')})


@app.route('/recommendations')
def recommendations():
    if not session.get('user_email'):
        return redirect(url_for('login'))
        
    predicted_mood = session.get('predicted_mood', 'happy')
    recommendations = recommend_songs(predicted_mood=predicted_mood)
    
    return render_template('recommendations.html', mood=predicted_mood, recommendations=recommendations)

@app.route('/live_mood', methods=['POST'])
def live_mood():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    
    predicted_mood = predict_camera_emotion(image_bytes)
    
    recommendations = recommend_songs(predicted_mood=predicted_mood)
    
    return jsonify({'mood': predicted_mood.capitalize(), 'recommendations': recommendations})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
