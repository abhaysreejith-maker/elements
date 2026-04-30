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
FACE_MESH_MODEL = None


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


def build_mood_profile(predicted_mood=None, mood_weights=None, top_emotions=None):
    profile = {}

    def add_weight(label, value):
        mood = normalize_mood_label(label)
        if not mood:
            return
        profile[mood] = profile.get(mood, 0.0) + float(value or 0.0)

    if isinstance(mood_weights, dict):
        for label, value in mood_weights.items():
            add_weight(label, value)
    elif isinstance(mood_weights, list):
        for item in mood_weights:
            if isinstance(item, dict):
                add_weight(item.get('emotion') or item.get('mood'), item.get('score', 0.0))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                add_weight(item[0], item[1])

    if isinstance(top_emotions, list):
        for item in top_emotions:
            if isinstance(item, dict):
                add_weight(item.get('emotion') or item.get('mood'), item.get('score', 0.0))

    if predicted_mood:
        add_weight(predicted_mood, 1.0)

    total = sum(profile.values())
    if total > 0:
        for label in list(profile.keys()):
            profile[label] = profile[label] / total

    return profile


def recommend_songs(predicted_mood, num_recommendations=10, mode='discover', user_email=None, mood_weights=None, top_emotions=None):
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
    mood_profile = build_mood_profile(predicted_mood=normalized_mood, mood_weights=mood_weights, top_emotions=top_emotions)
    if mood_profile:
        recommended_moods = [mood for mood, _ in sorted(mood_profile.items(), key=lambda item: item[1], reverse=True)]
    else:
        recommended_moods = mood_recommendations.get(normalized_mood, ['happy', 'calm'])
    time_bucket = get_time_bucket()

    profile_signature = tuple(sorted((mood, round(weight, 4)) for mood, weight in mood_profile.items())) if mood_profile else ()
    cache_key = (normalized_mood, normalized_mode, time_bucket, num_recommendations, profile_signature)
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

        # Blend all detected moods into the ranking so recommendations follow the average mood profile.
        filtered_songs['mood_score'] = 0.0
        if mood_profile:
            mood_lookup = filtered_songs['mood'].str.lower().map(lambda mood: float(mood_profile.get(mood, 0.0)))
            filtered_songs['mood_score'] = mood_lookup.fillna(0.0) * 100.0

        filtered_songs['final_score'] = filtered_songs['mode_score'] + filtered_songs['mood_score']

        if normalized_mode == 'discover':
            filtered_songs = filtered_songs.sort_values(by=['final_score', 'name'], ascending=[False, True], kind='mergesort')
        elif normalized_mode == 'familiar':
            filtered_songs = filtered_songs.sort_values(by=['final_score', 'artist'], ascending=[False, True], kind='mergesort')
        else:
            filtered_songs = filtered_songs.sort_values(by=['final_score', 'name', 'artist'], ascending=[False, True, True], kind='mergesort')

        songs = filtered_songs.drop_duplicates(subset=['name', 'artist']).head(num_recommendations)
        for _, song in songs.iterrows():
            recommended_songs.append({
                'name': song['name'],
                'artist': song['artist'],
                'mode': normalized_mode,
                'time_bucket': time_bucket
            })

    blended_mood = None
    if mood_profile:
        blended_moods = [mood for mood, weight in sorted(mood_profile.items(), key=lambda item: item[1], reverse=True) if weight > 0]
        blended_mood = ' / '.join(m.capitalize() for m in blended_moods[:3]) if blended_moods else None

    if blended_mood:
        print(f"Recommendation blend for {normalized_mood}: {blended_mood}")

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


# Try to import DeepFace; fall back to heuristic if unavailable
try:
    from deepface import DeepFace
except Exception:
    DeepFace = None

# Avoid startup hangs from heavy imports; use lazy loading for PIL
# cv2 and PIL are imported on-demand in predict_camera_emotion
cv2 = None

LAST_CAMERA_MOOD = 'neutral'
LAST_CAMERA_META = {
    'confidence': 0.0,
    'top_emotions': [],
    'latency_ms': 0
}

FACE_CASCADE = None
SMILE_CASCADE = None
FACE_MESH_MODEL = None


def heuristic_camera_emotion(img, started_at):
    """Analyze facial expression using Mediapipe face landmarks."""
    global LAST_CAMERA_MOOD, LAST_CAMERA_META, FACE_MESH_MODEL

    if img is None or len(img.shape) < 2:
        fallback = {
            'mood': LAST_CAMERA_MOOD,
            'confidence': max(30.0, float(LAST_CAMERA_META.get('confidence', 0.0))),
            'top_emotions': LAST_CAMERA_META.get('top_emotions') or [{'emotion': LAST_CAMERA_MOOD, 'score': 30.0}],
            'latency_ms': int((time.time() - started_at) * 1000),
            'landmarks': []
        }
        LAST_CAMERA_META = {
            'confidence': round(float(fallback['confidence']), 2),
            'top_emotions': fallback['top_emotions'],
            'latency_ms': fallback['latency_ms']
        }
        return fallback

    try:
        import mediapipe as mp

        if FACE_MESH_MODEL is None:
            FACE_MESH_MODEL = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )

        results = FACE_MESH_MODEL.process(img)
        if not results.multi_face_landmarks:
            LAST_CAMERA_MOOD = 'neutral'
            LAST_CAMERA_META = {
                'confidence': 25.0,
                'top_emotions': [{'emotion': 'neutral', 'score': 25.0}],
                'latency_ms': int((time.time() - started_at) * 1000)
            }
            return {
                'mood': 'neutral',
                'confidence': 25.0,
                'top_emotions': [{'emotion': 'neutral', 'score': 25.0}],
                'latency_ms': LAST_CAMERA_META['latency_ms'],
                'landmarks': []
            }

        landmarks = results.multi_face_landmarks[0].landmark
        height, width = img.shape[:2]

        def point(index):
            landmark = landmarks[index]
            return landmark.x, landmark.y

        def pixel_point(index):
            x, y = point(index)
            return {'x': int(x * width), 'y': int(y * height)}

        def distance(index_a, index_b):
            ax, ay = point(index_a)
            bx, by = point(index_b)
            return float(np.hypot(ax - bx, ay - by))

        left_eye_vertical = distance(159, 145)
        left_eye_horizontal = distance(33, 133)
        right_eye_vertical = distance(386, 374)
        right_eye_horizontal = distance(362, 263)
        avg_ear = ((left_eye_vertical / left_eye_horizontal) + (right_eye_vertical / right_eye_horizontal)) / 2.0 if left_eye_horizontal and right_eye_horizontal else 0.0

        mouth_vertical = distance(13, 14)
        mouth_horizontal = distance(61, 291)
        mouth_ratio = mouth_vertical / mouth_horizontal if mouth_horizontal else 0.0

        mouth_left_y = point(61)[1]
        mouth_right_y = point(291)[1]
        mouth_center_y = (point(13)[1] + point(14)[1]) / 2.0
        smile_lift = max(0.0, mouth_center_y - ((mouth_left_y + mouth_right_y) / 2.0))

        eye_center_y = ((point(159)[1] + point(145)[1]) / 2.0 + (point(386)[1] + point(374)[1]) / 2.0) / 2.0
        brow_left_y = point(70)[1]
        brow_right_y = point(300)[1]
        brow_gap = ((brow_left_y + brow_right_y) / 2.0) - eye_center_y

        scores = {
            'happy': 20.0,
            'sad': 18.0,
            'angry': 18.0,
            'neutral': 30.0,
            'calm': 20.0,
            'energetic': 20.0,
        }

        if avg_ear > 0.32:
            scores['energetic'] += 25.0
            scores['happy'] += 10.0
        elif avg_ear < 0.16:
            scores['sad'] += 20.0
            scores['calm'] += 10.0

        if mouth_ratio > 0.40:
            scores['energetic'] += 20.0
            scores['happy'] += 15.0
        elif mouth_ratio < 0.18:
            scores['calm'] += 10.0
            scores['neutral'] += 5.0

        if smile_lift > 0.015:
            scores['happy'] += 30.0
            scores['energetic'] += 10.0
            scores['neutral'] -= 5.0

        if brow_gap < 0.055:
            scores['angry'] += 25.0
            scores['sad'] += 5.0
        elif brow_gap > 0.11:
            scores['energetic'] += 10.0
            scores['happy'] += 5.0

        if mouth_ratio < 0.16 and avg_ear < 0.22 and abs(smile_lift) < 0.01:
            scores['calm'] += 20.0
            scores['neutral'] += 10.0

        mood = max(scores, key=scores.get)
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        confidence = round(min(95.0, max(35.0, sorted_scores[0][1])), 2)

        top_emotions = [
            {'emotion': emotion, 'score': round(max(8.0, min(100.0, value)), 2)}
            for emotion, value in sorted_scores[:3]
        ]

        LAST_CAMERA_MOOD = mood
        LAST_CAMERA_META = {
            'confidence': confidence,
            'top_emotions': top_emotions,
            'latency_ms': int((time.time() - started_at) * 1000)
        }
        return {
            'mood': mood,
            'confidence': confidence,
            'top_emotions': top_emotions,
            'latency_ms': LAST_CAMERA_META['latency_ms'],
            'landmarks': [pixel_point(i) for i in range(len(landmarks))]
        }
    except Exception:
        return {
            'mood': LAST_CAMERA_MOOD,
            'confidence': max(30.0, float(LAST_CAMERA_META.get('confidence', 0.0))),
            'top_emotions': LAST_CAMERA_META.get('top_emotions') or [{'emotion': LAST_CAMERA_MOOD, 'score': 30.0}],
            'latency_ms': int((time.time() - started_at) * 1000),
            'landmarks': []
        }


def predict_camera_emotion(image_bytes):
    """Detect emotion from camera image using PIL decoding and landmark analysis."""
    try:
        started_at = time.time()

        try:
            from PIL import Image
            import io as io_module
        except ImportError:
            return {
                'mood': LAST_CAMERA_MOOD,
                'confidence': LAST_CAMERA_META.get('confidence', 0.0),
                'top_emotions': LAST_CAMERA_META.get('top_emotions', []),
                'latency_ms': LAST_CAMERA_META.get('latency_ms', 0),
                'landmarks': []
            }

        try:
            pil_img = Image.open(io_module.BytesIO(image_bytes)).convert('RGB')
            img = np.array(pil_img)
        except Exception:
            img = None

        if img is None:
            return {
                'mood': LAST_CAMERA_MOOD,
                'confidence': LAST_CAMERA_META.get('confidence', 0.0),
                'top_emotions': LAST_CAMERA_META.get('top_emotions', []),
                'latency_ms': LAST_CAMERA_META.get('latency_ms', 0),
                'landmarks': []
            }

        return heuristic_camera_emotion(img, started_at)
    except Exception as exc:
        print(f'Error in emotion detection: {exc}')
        return {
            'mood': LAST_CAMERA_MOOD,
            'confidence': max(30.0, float(LAST_CAMERA_META.get('confidence', 0.0))),
            'top_emotions': LAST_CAMERA_META.get('top_emotions') or [{'emotion': LAST_CAMERA_MOOD, 'score': 30.0}],
            'latency_ms': LAST_CAMERA_META.get('latency_ms', 0),
            'landmarks': []
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
    top_emotions = camera_result.get('top_emotions', [])

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
        user_email=session.get('user_email'),
        top_emotions=top_emotions
    ) if include_recommendations else []

    blended_profile = build_mood_profile(predicted_mood=predicted_mood, top_emotions=top_emotions)
    blended_mood = ' / '.join(m.capitalize() for m in list(sorted(blended_profile, key=blended_profile.get, reverse=True))[:3]) if blended_profile else predicted_mood.capitalize()

    return jsonify({
        'mood': predicted_mood.capitalize(),
        'mood_key': predicted_mood.lower(),
        'blended_mood': blended_mood,
        'confidence': camera_result.get('confidence', 0.0),
        'top_emotions': camera_result.get('top_emotions', []),
        'latency_ms': camera_result.get('latency_ms', 0),
        'fusion_source': fusion.get('source', 'camera'),
        'fusion_confidence': fusion.get('source_confidence', camera_result.get('confidence', 0.0)),
        'recommendation_mode': recommendation_mode,
        'recommendations': recommendations_data,
        'recommendations_included': include_recommendations,
        'landmarks': camera_result.get('landmarks', [])
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
