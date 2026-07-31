import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Loaded model successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found. Please run train_model.py first.")

# Feature names expected by the model
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

@app.route('/')
def home():
    """Renders interactive web form for patient heart disease risk prediction."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for cloud deployment (Render)."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Task 3: REST API endpoint for heart disease prediction.
    Accepts JSON input or Form input, returns JSON prediction response.
    """
    if model is None:
        return jsonify({'error': 'Model artifact not loaded. Train the model first.'}), 500
    
    try:
        data = None
        # Handle JSON input or Form submission input
        if request.is_json:
            data = request.get_json()
        elif request.form:
            data = request.form.to_dict()
        else:
            return jsonify({'error': 'Invalid request format. Send JSON or form data.'}), 400
        
        # Extract features and validate presence
        input_features = []
        for feature in FEATURE_NAMES:
            if feature not in data:
                return jsonify({
                    'error': f'Missing required feature parameter: {feature}',
                    'expected_features': FEATURE_NAMES
                }), 400
            
            # Cast values to float
            val = float(data[feature])
            input_features.append(val)
        
        # Format input array for model prediction
        input_array = np.array(input_features).reshape(1, -1)
        df_input = pd.DataFrame(input_array, columns=FEATURE_NAMES)
        
        # Make prediction
        prediction_val = int(model.predict(df_input)[0])
        
        # Get confidence/probability if supported by model
        confidence = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df_input)[0]
            confidence = round(float(probabilities[prediction_val]), 4)
            
        prediction_label = "Heart Disease Detected" if prediction_val == 1 else "No Heart Disease Detected"
        
        # Standardized API response
        response = {
            "prediction": prediction_label,
            "prediction_code": prediction_val,
            "confidence": confidence,
            "status": "success",
            "input_received": {k: float(data[k]) for k in FEATURE_NAMES}
        }
        
        # If request came from web UI HTML form, render template with result
        if not request.is_json and request.form:
            return render_template('index.html', result=response, input_data=data)
            
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
