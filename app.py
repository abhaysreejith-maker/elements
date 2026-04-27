from flask import Flask, render_template, redirect, request, url_for, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import os
import base64
import time
from datetime import datetime

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


class MockModel:
    def predict(self, features):
        res = np.random.rand(1, 8)
        res = res / res.sum()
        return res


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'static/audio'


@app.context_processor
def inject_user():
    return dict(logged_in_user=session.get('user_email'))


def get_db_connection():
    return sqlite3.connect('music_rec.db')


def executionquery(query, values):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, values)
    conn.commit()
    conn.close()


def retrivequery1(query, values):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, values)
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


def normalize_mood_label(value):
    mapping = {
        'fear': 'fearful',
        'fearful': 'fearful',
        'surprise': 'surprised',
        'surprised': 'surprised',
        'happiness': 'happy',
        'happy': 'happy',
        'sadness': 'sad',
        'sad': 'sad',
        'anger': 'angry',
        'angry': 'angry',
        'disgust': 'disgust',
        'neutral': 'neutral'
    }
    return mapping.get(str(value or '').strip().lower(), 'neutral')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/About')
def about():
    return render_template('about.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['useremail']
        password = request.form['password']
        c_password = request.form['c_password']
        username = request.form['username']
        age = request.form['age']
        gender = request.form['gender']
        if password == c_password:
            query = 'SELECT UPPER(email) FROM users'
            email_data = retrivequery2(query)
            email_data_list = [item[0] for item in email_data]
            if email.upper() not in email_data_list:
                hashed_password = generate_password_hash(password)
                query = 'INSERT INTO users (email, password, username, age, gender) VALUES (?, ?, ?, ?, ?)'
                values = (email, hashed_password, username, age, gender)
                executionquery(query, values)
                return render_template('login.html', message='Successfully registered. Please sign in.')
            return render_template('register.html', message='This email is already registered.')
        return render_template('register.html', message='Confirm password does not match.')
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['useremail']
        password = request.form['password']

        query = 'SELECT password FROM users WHERE UPPER(email) = ?'
        values = (email.upper(),)
        password_data = retrivequery1(query, values)

        if password_data and check_password_hash(password_data[0][0], password):
            session['user_email'] = email
            return redirect(url_for('home'))
        return render_template('login.html', message='Invalid email or password.')
    return render_template('login.html')


@app.route('/home')
def home():
    if not session.get('user_email'):
        return redirect(url_for('login'))
    return render_template('home.html')


@app.route('/insights')
def insights():
    if not session.get('user_email'):
        return redirect(url_for('login'))

    email = session.get('user_email')
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        'SELECT mood, source, confidence, recommendation_mode, created_at FROM mood_events WHERE email = ? ORDER BY id DESC LIMIT 80',
        (email,)
    )
    events = [dict(row) for row in cursor.fetchall()]

    cursor.execute(
        'SELECT verdict, COUNT(*) as count FROM mood_feedback WHERE email = ? GROUP BY verdict',
        (email,)
    )
    feedback_rows = cursor.fetchall()
    conn.close()

    feedback_summary = {'match': 0, 'mismatch': 0}
    for row in feedback_rows:
        feedback_summary[str(row['verdict'])] = int(row['count'])

    mood_counts = {}
    for item in events:
        mood = str(item.get('mood', 'neutral'))
        mood_counts[mood] = mood_counts.get(mood, 0) + 1
    dominant_mood = sorted(mood_counts.items(), key=lambda x: x[1], reverse=True)[0][0] if mood_counts else 'neutral'

    return render_template(
        'insights.html',
        events=events,
        feedback_summary=feedback_summary,
        dominant_mood=dominant_mood,
        total_events=len(events)
    )


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/demo_login')
def demo_login():
    session['user_email'] = 'demo@example.com'
    return redirect(url_for('home'))


try:
    data = pd.read_csv('data_moods.csv')
    features = ['danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'valence', 'loudness', 'speechiness', 'tempo']
    feature_matrix = data[features]
    scaler = StandardScaler()
    normalized_matrix = scaler.fit_transform(feature_matrix)
    svd = TruncatedSVD(n_components=2)
    decomposed_matrix = svd.fit_transform(normalized_matrix)
    svd_matrix = np.dot(decomposed_matrix, svd.components_)
    similarity_matrix = cosine_similarity(svd_matrix)
except Exception as e:
    print(f'Error loading CSV or processing features: {e}')
    data = pd.DataFrame()
    similarity_matrix = np.array([])


RECOMMENDATION_CACHE = {}
LAST_USER_EVENT = {}


def ensure_runtime_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS mood_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            mood TEXT,
            recommendation_mode TEXT,
            verdict TEXT,
            created_at TEXT
        )
        '''
    )
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS mood_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            mood TEXT,
            source TEXT,
            confidence REAL,
            recommendation_mode TEXT,
            created_at TEXT
        )
        '''
    )
    conn.commit()
    conn.close()


def get_time_bucket():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return 'morning'
    if 12 <= hour < 17:
        return 'afternoon'
    if 17 <= hour < 22:
        return 'evening'
    return 'night'


def recommendation_profile(mode, time_bucket):
    mode_profiles = {
        'discover': {'energy': 0.18, 'valence': 0.12, 'danceability': 0.08},
        'familiar': {'energy': 0.05, 'valence': 0.08, 'danceability': 0.05},
        'focus': {'instrumentalness': 0.22, 'speechiness': -0.2, 'energy': -0.08},
        'chill': {'acousticness': 0.18, 'energy': -0.15, 'loudness': -0.1}
    }
    time_profiles = {
        'morning': {'energy': 0.06, 'valence': 0.06},
        'afternoon': {'danceability': 0.05},
        'evening': {'acousticness': 0.08, 'energy': -0.05},
        'night': {'instrumentalness': 0.08, 'loudness': -0.1}
    }

    profile = {}
    for source in (mode_profiles.get(mode, {}), time_profiles.get(time_bucket, {})):
        for key, value in source.items():
            profile[key] = profile.get(key, 0.0) + value
    return profile


def recommend_songs(predicted_mood, num_recommendations=10, mode='discover', user_email=None):
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

    normalized_mood = normalize_mood_label(predicted_mood)
    normalized_mode = str(mode or 'discover').lower()
    recommended_moods = mood_recommendations.get(normalized_mood, ['happy', 'calm'])
    time_bucket = get_time_bucket()

    cache_key = (normalized_mood, normalized_mode, time_bucket, num_recommendations)
    if cache_key in RECOMMENDATION_CACHE:
        return RECOMMENDATION_CACHE[cache_key]

    recommended_songs = []
    if similarity_matrix.size == 0 or data.empty:
        return []

    filtered_songs = data[data['mood'].str.lower().isin(recommended_moods)].copy()
    if not filtered_songs.empty:
        profile = recommendation_profile(normalized_mode, time_bucket)
        filtered_songs['mode_score'] = 0.0
        for column, weight in profile.items():
            if column in filtered_songs.columns:
                filtered_songs['mode_score'] += filtered_songs[column] * float(weight)

        if normalized_mode == 'discover':
            filtered_songs = filtered_songs.sort_values(by=['mode_score', 'name'], ascending=[False, True], kind='mergesort')
        elif normalized_mode == 'familiar':
            filtered_songs = filtered_songs.sort_values(by=['mode_score', 'artist'], ascending=[False, True], kind='mergesort')
        else:
            filtered_songs = filtered_songs.sort_values(by=['mode_score', 'name', 'artist'], ascending=[False, True, True], kind='mergesort')

        songs = filtered_songs.drop_duplicates(subset=['name', 'artist']).head(num_recommendations)
        for _, song in songs.iterrows():
            recommended_songs.append({
                'name': song['name'],
                'artist': song['artist'],
                'mode': normalized_mode,
                'time_bucket': time_bucket
            })

    RECOMMENDATION_CACHE[cache_key] = recommended_songs
    return recommended_songs


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('user_email'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'audio' not in request.files:
            return render_template('upload.html', message='No file selected.')

        myfile = request.files['audio']
        if myfile.filename == '':
            return render_template('upload.html', message='No file selected.')

        accepted_formats = ['mp3', 'wav', 'ogg', 'flac']
        if myfile.filename.split('.')[-1].lower() not in accepted_formats:
            message = 'Invalid file format. Accepted formats: {}'.format(', '.join(accepted_formats))
            return render_template('upload.html', message=message)

        filename = secure_filename(myfile.filename)
        mypath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        myfile.save(mypath)

        result = record_audio(record=False, file_loc=mypath)
        recommendations = recommend_songs(predicted_mood=result)
        return render_template('upload.html', prediction=result, path=mypath, recommendations=recommendations)

    return render_template('upload.html')


CHUNK = 1024 * 4
FORMAT = 8
CHANNELS = 1
RATE = 48000
model = MockModel()


def extract_features(file_name):
    try:
        x_data, sample_rate = librosa.load(file_name)
        stft = np.abs(librosa.stft(x_data))
        mfccs = np.mean(librosa.feature.mfcc(y=x_data, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=x_data, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(x_data), sr=sample_rate).T, axis=0)
        return mfccs, chroma, mel, contrast, tonnetz
    except Exception as e:
        print(f'Feature extraction error: {e}')
        return np.zeros(40), np.zeros(12), np.zeros(128), np.zeros(7), np.zeros(6)


def speech_to_emotion(filename):
    mfccs, chroma, mel, contrast, tonnetz = extract_features(filename)
    features = np.hstack([mfccs, chroma, mel, contrast, tonnetz]).reshape(1, -1)
    probabilities = model.predict(features)
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgused', 'surprised']
    idx = np.argmax(probabilities, axis=1)
    prob = np.max(probabilities, axis=1)
    return emotions[idx[0]], prob[0]


def record_audio(record=True, file_loc=None):
    if record:
        if pyaudio is None:
            return 'Microphone recording not available (pyaudio missing)'
        return 'Recording finished (Demo)'
    if file_loc:
        return speech_to_emotion(file_loc)[0].capitalize()
    return 'Neutral'


try:
    from deepface import DeepFace
except Exception:
    DeepFace = None
import cv2

LAST_CAMERA_MOOD = 'neutral'
LAST_CAMERA_META = {
    'confidence': 0.0,
    'top_emotions': []
}


def predict_camera_emotion(image_bytes):
    global LAST_CAMERA_MOOD, LAST_CAMERA_META
    try:
        if DeepFace is None:
            return {
                'mood': LAST_CAMERA_MOOD,
                'confidence': LAST_CAMERA_META.get('confidence', 0.0),
                'top_emotions': LAST_CAMERA_META.get('top_emotions', []),
                'latency_ms': LAST_CAMERA_META.get('latency_ms', 0)
            }

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {
                'mood': LAST_CAMERA_MOOD,
                'confidence': LAST_CAMERA_META.get('confidence', 0.0),
                'top_emotions': LAST_CAMERA_META.get('top_emotions', []),
                'latency_ms': LAST_CAMERA_META.get('latency_ms', 0)
            }

        height, width = img.shape[:2]
        max_width = 320
        if width > max_width:
            scale = max_width / float(width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

        started_at = time.time()
        result = DeepFace.analyze(
            img,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )

        payload = result[0] if isinstance(result, list) and result else result if isinstance(result, dict) else {}
        dominant_emotion = payload.get('dominant_emotion', 'neutral')
        emotion_scores = payload.get('emotion', {}) or {}

        final_mood = normalize_mood_label(dominant_emotion)
        neutral_score = float(emotion_scores.get('neutral', 0.0))
        happy_score = float(emotion_scores.get('happy', 0.0))
        sad_score = float(emotion_scores.get('sad', 0.0))
        angry_score = float(emotion_scores.get('angry', 0.0))

        if final_mood in {'fearful', 'surprised'} and (neutral_score >= 30.0 or max(happy_score, sad_score, angry_score) >= 25.0):
            final_mood = 'neutral'

        dominant_score = float(emotion_scores.get(dominant_emotion, 0.0))
        if dominant_score < 18.0:
            final_mood = 'neutral'

        if happy_score >= 35.0 and happy_score > neutral_score:
            final_mood = 'happy'
        elif sad_score >= 35.0 and sad_score > neutral_score:
            final_mood = 'sad'
        elif angry_score >= 35.0 and angry_score > neutral_score:
            final_mood = 'angry'
        elif neutral_score >= 40.0:
            final_mood = 'neutral'

        sorted_emotions = sorted(
            [(str(key), float(value)) for key, value in emotion_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        top_emotions = [{'emotion': emotion, 'score': round(score, 2)} for emotion, score in sorted_emotions[:3]]
        confidence = float(emotion_scores.get(dominant_emotion, dominant_score))

        LAST_CAMERA_MOOD = final_mood
        LAST_CAMERA_META = {
            'confidence': round(confidence, 2),
            'top_emotions': top_emotions,
            'latency_ms': int((time.time() - started_at) * 1000)
        }
        return {
            'mood': final_mood,
            'confidence': LAST_CAMERA_META['confidence'],
            'top_emotions': LAST_CAMERA_META['top_emotions'],
            'latency_ms': LAST_CAMERA_META['latency_ms']
        }
    except Exception as e:
        print(f'Error in emotion detection: {e}')
        return {
            'mood': LAST_CAMERA_MOOD,
            'confidence': LAST_CAMERA_META.get('confidence', 0.0),
            'top_emotions': LAST_CAMERA_META.get('top_emotions', []),
            'latency_ms': LAST_CAMERA_META.get('latency_ms', 0)
        }


def fuse_moods(camera_mood, camera_confidence, user_hint_mood):
    normalized_camera = normalize_mood_label(camera_mood)
    normalized_hint = normalize_mood_label(user_hint_mood)
    confidence = float(camera_confidence or 0.0)

    if normalized_hint == 'neutral':
        return {'mood': normalized_camera, 'source': 'camera', 'source_confidence': round(confidence, 2)}

    if confidence < 42.0:
        return {'mood': normalized_hint, 'source': 'camera+checkin', 'source_confidence': round(max(confidence, 42.0), 2)}

    if normalized_camera != normalized_hint and confidence < 56.0:
        return {'mood': normalized_hint, 'source': 'camera+checkin', 'source_confidence': round(confidence, 2)}

    return {'mood': normalized_camera, 'source': 'camera', 'source_confidence': round(confidence, 2)}


def store_mood_event(email, mood, source, confidence, recommendation_mode):
    if not email:
        return

    state = LAST_USER_EVENT.get(email)
    now_ts = time.time()
    should_store = state is None or state.get('mood') != mood or (now_ts - float(state.get('at', 0.0))) > 20.0
    if not should_store:
        return

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO mood_events (email, mood, source, confidence, recommendation_mode, created_at) VALUES (?, ?, ?, ?, ?, ?)',
        (email, mood, source, float(confidence or 0.0), recommendation_mode, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
    LAST_USER_EVENT[email] = {'mood': mood, 'at': now_ts}


@app.route('/capture_mood', methods=['POST'])
def capture_mood():
    data = request.get_json(silent=True) or {}
    image_raw = data.get('image', '')
    if ',' not in image_raw:
        return jsonify({'error': 'Invalid image payload'}), 400

    image_bytes = base64.b64decode(image_raw.split(',', 1)[1])
    camera_result = predict_camera_emotion(image_bytes)
    session['predicted_mood'] = camera_result.get('mood', 'neutral')
    return jsonify({'redirect': url_for('recommendations')})


@app.route('/recommendations')
def recommendations():
    if not session.get('user_email'):
        return redirect(url_for('login'))

    predicted_mood = session.get('predicted_mood', 'happy')
    recommendations_data = recommend_songs(predicted_mood=predicted_mood, mode='discover', user_email=session.get('user_email'))
    return render_template('recommendations.html', mood=predicted_mood, recommendations=recommendations_data)


@app.route('/live_mood', methods=['POST'])
def live_mood():
    data = request.get_json(silent=True) or {}
    image_raw = data.get('image', '')
    if ',' not in image_raw:
        return jsonify({'error': 'Invalid image payload'}), 400

    image_bytes = base64.b64decode(image_raw.split(',', 1)[1])
    include_recommendations = bool(data.get('include_recommendations', True))
    recommendation_mode = str(data.get('recommendation_mode', 'discover')).lower()
    user_hint_mood = str(data.get('user_hint_mood', 'neutral')).lower()

    camera_result = predict_camera_emotion(image_bytes)
    fusion = fuse_moods(camera_result.get('mood', 'neutral'), camera_result.get('confidence', 0.0), user_hint_mood)
    predicted_mood = fusion.get('mood', 'neutral')

    store_mood_event(
        session.get('user_email'),
        predicted_mood,
        fusion.get('source', 'camera'),
        fusion.get('source_confidence', camera_result.get('confidence', 0.0)),
        recommendation_mode
    )

    recommendations_data = recommend_songs(
        predicted_mood=predicted_mood,
        mode=recommendation_mode,
        user_email=session.get('user_email')
    ) if include_recommendations else []

    return jsonify({
        'mood': predicted_mood.capitalize(),
        'mood_key': predicted_mood.lower(),
        'confidence': camera_result.get('confidence', 0.0),
        'top_emotions': camera_result.get('top_emotions', []),
        'latency_ms': camera_result.get('latency_ms', 0),
        'fusion_source': fusion.get('source', 'camera'),
        'fusion_confidence': fusion.get('source_confidence', camera_result.get('confidence', 0.0)),
        'recommendation_mode': recommendation_mode,
        'recommendations': recommendations_data,
        'recommendations_included': include_recommendations
    })


@app.route('/mood_feedback', methods=['POST'])
def mood_feedback():
    if not session.get('user_email'):
        return jsonify({'error': 'Unauthorized'}), 401

    payload = request.get_json(silent=True) or {}
    verdict = str(payload.get('verdict', '')).lower()
    mood = normalize_mood_label(payload.get('mood', 'neutral'))
    recommendation_mode = str(payload.get('recommendation_mode', 'discover')).lower()

    if verdict not in {'match', 'mismatch'}:
        return jsonify({'error': 'Invalid verdict'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO mood_feedback (email, mood, recommendation_mode, verdict, created_at) VALUES (?, ?, ?, ?, ?)',
        (session.get('user_email'), mood, recommendation_mode, verdict, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    ensure_runtime_tables()
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode, use_reloader=False, port=5001)
