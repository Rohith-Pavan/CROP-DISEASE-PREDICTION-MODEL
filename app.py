import os
import json
import sqlite3
import numpy as np
import time
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess, decode_predictions
from PIL import Image
import io
import ssl
import certifi

# Fix SSL on Mac
os.environ['SSL_CERT_FILE'] = certifi.where()

app = Flask(__name__)

# Configuration
MODEL_PATH = 'saved_models/resnet50_model.h5'
METADATA_PATH = 'saved_models/metadata.json'
RECOMMENDATIONS_PATH = 'saved_models/disease_recommendations.json'
UPLOAD_FOLDER = 'static/uploads'
DB_PATH = 'history.db'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
model = None
ood_model = None
class_names = []
recommendations = {}

# ==========================================
# DATABASE SETUP
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            disease_name TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_crop BOOLEAN
        )
    ''')
    conn.commit()
    conn.close()

# ==========================================
# MODEL LOADER
# ==========================================
def load_data():
    global model, ood_model, class_names, recommendations
    
    print("⏳ Loading main ResNet50 model...")
    try:
        model = load_model(MODEL_PATH)
        print("✅ Main model loaded.")
    except Exception as e:
        print(f"❌ Error loading main model: {e}")
    
    print("⏳ Loading MobileNetV2 (OOD Detector)...")
    try:
        ood_model = MobileNetV2(weights='imagenet')
        print("✅ MobileNetV2 loaded.")
    except Exception as e:
        print(f"❌ Error loading MobileNetV2: {e}")

    print(f"⏳ Loading metadata...")
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
            class_names = metadata.get('classes', [])
        print(f"✅ Loaded {len(class_names)} classes.")
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")

    try:
        with open(RECOMMENDATIONS_PATH, 'r') as f:
            recommendations = json.load(f)
        print("✅ Recommendations loaded.")
    except Exception as e:
        print(f"❌ Error loading recommendations: {e}")

# Initialize
init_db()
load_data()

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_relevant_image(img_array):
    """
    Uses MobileNetV2 to check if the image matches general plant-related categories.
    """
    if ood_model is None:
        return True, "OOD Model not loaded"
        
    # Preprocess for MobileNetV2
    # img_array is (1, 224, 224, 3) range 0-255 (from keras load_img)
    # MobileNetV2 expects -1 to 1 or specific preprocessing
    # We need to make sure we don't double process if we passed pre-processed data
    
    # Let's assume we pass raw array here (0-255)
    mobilenet_input = mobilenet_preprocess(img_array.copy())
    preds = ood_model.predict(mobilenet_input)
    decoded = decode_predictions(preds, top=5)[0]
    
    # Keywords indicating a plant/crop context
    plant_keywords = [
        'plant', 'tree', 'flower', 'fruit', 'vegetable', 'leaf', 'grass', 'garden', 
        'agriculture', 'pot', 'greenhouse', 'corn', 'maize', 'wheat', 'crop', 
        'mushroom', 'fungus', 'broccoli', 'cabbage', 'cauliflower', 'zucchini',
        'cucumber', 'tomato', 'potato', 'strawberry', 'apple', 'orange', 'lemon',
        'banana', 'pomegranate', 'grape', 'fig', 'pineapple', 'lettuce', 'spinach',
        # Pests and Insects
        'insect', 'bug', 'beetle', 'worm', 'larva', 'caterpillar', 'slug', 'snail',
        'fly', 'ant', 'cricket', 'grasshopper', 'moth', 'weevil', 'pest', 'spider',
        'locust', 'mantis', 'bee', 'wasp'
    ]
    
    # Check top 5 predictions
    is_plant = False
    detected_labels = []
    
    for _, label, score in decoded:
        detected_labels.append(label)
        # Check if label contains any keyword
        label_lower = label.lower()
        if any(keyword in label_lower for keyword in plant_keywords):
            is_plant = True
            break
            
    return is_plant, detected_labels

# ==========================================
# ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        try:
            # Save file
            filename = secure_filename(f"{int(time.time())}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process image for Prediction
            img = keras_image.load_img(filepath, target_size=(224, 224))
            img_array = keras_image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            
            # 1. OOD Check
            is_crop, labels = is_relevant_image(img_batch)
            
            if not is_crop:
                # Log to DB as rejected
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute('INSERT INTO history (filename, disease_name, confidence, is_crop) VALUES (?, ?, ?, ?)',
                          (filename, "Not a Crop", 0.0, False))
                conn.commit()
                conn.close()
                
                return jsonify({
                    'is_crop': False,
                    'message': f"We detected this isn't a crop image (Matched: {', '.join(labels[:2])}). Please upload a clear plant leaf image.",
                    'image_url': f"/{UPLOAD_FOLDER}/{filename}"
                })

            # 2. Disease Prediction
            resnet_input = resnet_preprocess(img_batch.copy())
            predictions = model.predict(resnet_input)
            predicted_class_idx = np.argmax(predictions[0])
            confidence_score = float(predictions[0][predicted_class_idx])
            predicted_class_name = class_names[predicted_class_idx]

            # Get details
            disease_info = recommendations.get(predicted_class_name, {})
            if not disease_info:
                 disease_info = {
                    "disease_name": predicted_class_name.replace('_', ' ').title(),
                    "description": "No specific details available.",
                    "symptoms": [], "management_steps": [], "prevention": []
                }

            # Response
            response = {
                'is_crop': True,
                'class_name': predicted_class_name,
                'confidence': confidence_score,
                'details': disease_info,
                'image_url': f"/{UPLOAD_FOLDER}/{filename}"
            }
            
            # Log to DB
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('INSERT INTO history (filename, disease_name, confidence, is_crop) VALUES (?, ?, ?, ?)',
                      (filename, predicted_class_name, confidence_score, True))
            conn.commit()
            conn.close()
            
            return jsonify(response)

        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/api/history')
def get_history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM history ORDER BY timestamp DESC LIMIT 50')
    rows = c.fetchall()
    conn.close()
    
    history_data = []
    for row in rows:
        history_data.append({
            'id': row['id'],
            'image_url': f"/{UPLOAD_FOLDER}/{row['filename']}",
            'disease_name': row['disease_name'],
            'confidence': row['confidence'],
            'is_crop': bool(row['is_crop']),
            'date': row['timestamp']
        })
    
    return jsonify(history_data)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
