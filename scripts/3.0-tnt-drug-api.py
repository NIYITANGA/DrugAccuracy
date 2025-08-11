"""
Drug Prediction API - Version 3.0
Author: TNT
Description: REST API for drug prediction based on patient characteristics
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and encoders
model = None
encoders = None

def load_model_and_encoders():
    """Load the trained model and encoders"""
    global model, encoders
    
    try:
        model_path = '../output/random_forest_drug_model.pkl'
        encoders_path = '../output/label_encoders.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(encoders_path):
            logger.error("Model files not found. Please run 1.0-tnt-drug-prediction.py first.")
            return False
            
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        
        logger.info("Model and encoders loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def validate_input(data):
    """Validate input data"""
    required_fields = ['age', 'sex', 'bp', 'cholesterol', 'na_to_k']
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate age
    try:
        age = int(data['age'])
        if age < 0 or age > 120:
            return False, "Age must be between 0 and 120"
    except (ValueError, TypeError):
        return False, "Age must be a valid integer"
    
    # Validate sex
    if data['sex'].upper() not in ['M', 'F']:
        return False, "Sex must be 'M' or 'F'"
    
    # Validate blood pressure
    if data['bp'].upper() not in ['LOW', 'NORMAL', 'HIGH']:
        return False, "BP must be 'LOW', 'NORMAL', or 'HIGH'"
    
    # Validate cholesterol
    if data['cholesterol'].upper() not in ['NORMAL', 'HIGH']:
        return False, "Cholesterol must be 'NORMAL' or 'HIGH'"
    
    # Validate Na_to_K ratio
    try:
        na_to_k = float(data['na_to_k'])
        if na_to_k < 0 or na_to_k > 50:
            return False, "Na_to_K ratio must be between 0 and 50"
    except (ValueError, TypeError):
        return False, "Na_to_K must be a valid number"
    
    return True, "Valid input"

def predict_drug(age, sex, bp, cholesterol, na_to_k):
    """Make drug prediction"""
    try:
        # Encode categorical variables
        sex_encoded = encoders['Sex'].transform([sex.upper()])[0]
        bp_encoded = encoders['BP'].transform([bp.upper()])[0]
        chol_encoded = encoders['Cholesterol'].transform([cholesterol.upper()])[0]
        
        # Create feature array
        features = np.array([[age, sex_encoded, bp_encoded, chol_encoded, na_to_k]])
        
        # Make prediction
        prediction_encoded = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        # Decode prediction
        predicted_drug = encoders['Drug'].inverse_transform([prediction_encoded])[0]
        confidence = float(max(prediction_proba))
        
        # Get all class probabilities
        drug_classes = encoders['Drug'].classes_
        probabilities = {}
        for i, drug_class in enumerate(drug_classes):
            probabilities[drug_class] = float(prediction_proba[i])
        
        return {
            'predicted_drug': predicted_drug,
            'confidence': confidence,
            'all_probabilities': probabilities
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return None

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Drug Prediction API - Version 3.0',
        'author': 'TNT',
        'description': 'REST API for predicting drug outcomes based on patient characteristics',
        'endpoints': {
            'POST /predict': 'Make drug prediction',
            'GET /health': 'Check API health',
            'GET /model-info': 'Get model information'
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_loaded = model is not None and encoders is not None
    
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None or encoders is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'Random Forest Classifier',
        'accuracy': '97.5%',
        'features': ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],
        'target_classes': list(encoders['Drug'].classes_),
        'feature_mappings': {
            'Sex': dict(zip(encoders['Sex'].classes_, encoders['Sex'].transform(encoders['Sex'].classes_))),
            'BP': dict(zip(encoders['BP'].classes_, encoders['BP'].transform(encoders['BP'].classes_))),
            'Cholesterol': dict(zip(encoders['Cholesterol'].classes_, encoders['Cholesterol'].transform(encoders['Cholesterol'].classes_)))
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None or encoders is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure model files are available.',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Validate input
        is_valid, message = validate_input(data)
        if not is_valid:
            return jsonify({
                'error': f'Invalid input: {message}',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Extract and convert data
        age = int(data['age'])
        sex = data['sex']
        bp = data['bp']
        cholesterol = data['cholesterol']
        na_to_k = float(data['na_to_k'])
        
        # Make prediction
        result = predict_drug(age, sex, bp, cholesterol, na_to_k)
        
        if result is None:
            return jsonify({
                'error': 'Prediction failed',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        # Return successful prediction
        response = {
            'success': True,
            'input': {
                'age': age,
                'sex': sex.upper(),
                'bp': bp.upper(),
                'cholesterol': cholesterol.upper(),
                'na_to_k': na_to_k
            },
            'prediction': {
                'recommended_drug': result['predicted_drug'],
                'confidence': round(result['confidence'] * 100, 2),
                'confidence_percentage': f"{round(result['confidence'] * 100, 1)}%"
            },
            'all_drug_probabilities': {
                drug: f"{round(prob * 100, 1)}%" 
                for drug, prob in result['all_probabilities'].items()
            },
            'timestamp': datetime.now().isoformat(),
            'disclaimer': 'This prediction is for educational purposes only and should not be used for actual medical decisions without proper clinical validation.'
        }
        
        logger.info(f"Successful prediction for patient: Age={age}, Sex={sex}, BP={bp}, Cholesterol={cholesterol}, Na_to_K={na_to_k}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint for multiple patients"""
    try:
        # Check if model is loaded
        if model is None or encoders is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure model files are available.',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'patients' not in data:
            return jsonify({
                'error': 'No patient data provided. Expected format: {"patients": [patient_data]}',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        patients = data['patients']
        if not isinstance(patients, list):
            return jsonify({
                'error': 'Patients data must be a list',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        results = []
        errors = []
        
        for i, patient_data in enumerate(patients):
            # Validate input
            is_valid, message = validate_input(patient_data)
            if not is_valid:
                errors.append({
                    'patient_index': i,
                    'error': f'Invalid input: {message}',
                    'patient_data': patient_data
                })
                continue
            
            # Extract and convert data
            age = int(patient_data['age'])
            sex = patient_data['sex']
            bp = patient_data['bp']
            cholesterol = patient_data['cholesterol']
            na_to_k = float(patient_data['na_to_k'])
            
            # Make prediction
            result = predict_drug(age, sex, bp, cholesterol, na_to_k)
            
            if result is None:
                errors.append({
                    'patient_index': i,
                    'error': 'Prediction failed',
                    'patient_data': patient_data
                })
                continue
            
            # Add successful prediction
            results.append({
                'patient_index': i,
                'input': {
                    'age': age,
                    'sex': sex.upper(),
                    'bp': bp.upper(),
                    'cholesterol': cholesterol.upper(),
                    'na_to_k': na_to_k
                },
                'prediction': {
                    'recommended_drug': result['predicted_drug'],
                    'confidence': round(result['confidence'] * 100, 2),
                    'confidence_percentage': f"{round(result['confidence'] * 100, 1)}%"
                }
            })
        
        response = {
            'success': True,
            'total_patients': len(patients),
            'successful_predictions': len(results),
            'failed_predictions': len(errors),
            'results': results,
            'errors': errors,
            'timestamp': datetime.now().isoformat(),
            'disclaimer': 'These predictions are for educational purposes only and should not be used for actual medical decisions without proper clinical validation.'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch API error: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Please check the API documentation for available endpoints',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'Please check the API documentation for correct HTTP methods',
        'timestamp': datetime.now().isoformat()
    }), 405

if __name__ == '__main__':
    print("="*60)
    print("DRUG PREDICTION API - VERSION 3.0")
    print("="*60)
    
    # Load model and encoders
    if load_model_and_encoders():
        print("✅ Model loaded successfully!")
        print("\nAPI Endpoints:")
        print("- GET  /           : API information")
        print("- GET  /health     : Health check")
        print("- GET  /model-info : Model information")
        print("- POST /predict    : Single prediction")
        print("- POST /predict-batch : Batch predictions")
        
        print(f"\n🚀 Starting API server...")
        print("📡 API will be available at: http://localhost:5000")
        print("📖 API documentation at: http://localhost:5000")
        print("\n⚠️  Disclaimer: This API is for educational purposes only.")
        print("   Do not use for actual medical decisions without clinical validation.")
        print("="*60)
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please run 1.0-tnt-drug-prediction.py first.")
        print("   Make sure the following files exist:")
        print("   - ../output/random_forest_drug_model.pkl")
        print("   - ../output/label_encoders.pkl")
