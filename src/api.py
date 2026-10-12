import os
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Resolution du chemin du modele
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'iris_model.pkl')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join('models', 'iris_model.pkl')

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

MODEL_VERSION = "1.0.0"

@app.get('/')
def home():
    return jsonify({
        'service': 'Iris MLOps API',
        'status': 'running',
        'model_version': MODEL_VERSION
    })

# Challenge 11 : Endpoint /health
@app.get('/health')
def health():
    is_ready = model is not None
    return jsonify({
        'status': 'healthy' if is_ready else 'unhealthy',
        'model_loaded': is_ready,
        'model_version': MODEL_VERSION
    }), 200 if is_ready else 503

@app.post('/predict')
def predict():
    if model is None:
        return jsonify({'error': 'Modele non charge'}), 500
    
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'Format attendu: {"features": [sepal_l, sepal_w, petal_l, petal_w]}'}), 400
    
    features = data['features']
    prediction = model.predict([features])
    return jsonify({
        'prediction': int(prediction[0]),
        'model_version': MODEL_VERSION
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
