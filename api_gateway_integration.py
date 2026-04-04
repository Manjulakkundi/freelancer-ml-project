from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Import your trained models
sys.path.append('.')
from train_success_model import FreelancerSuccessPredictor
from train_fake_detector import FakeProfileDetector

# Create Flask app
app = Flask(__name__)
CORS(app)

# Load models when server starts
print("🔄 Loading ML models...")
success_predictor = FreelancerSuccessPredictor()
success_predictor.load_model('success_predictor.pkl')

fake_detector = FakeProfileDetector()
fake_detector.load_model('fake_detector.pkl')
print("✅ Models loaded!\n")


def normalize_parameters(params, model_type):
    """
    Normalize parameters from AI Gateway format to ML model format
    """
    normalized = params.copy()
    
    if model_type == "success_prediction":
        if 'completion_rate' in normalized and normalized['completion_rate'] > 1:
            normalized['completion_rate'] /= 100.0
        
        if 'on_time_delivery_rate' in normalized and normalized['on_time_delivery_rate'] > 1:
            normalized['on_time_delivery_rate'] /= 100.0
        
        if 'skill_match_score' in normalized and normalized['skill_match_score'] > 1:
            normalized['skill_match_score'] /= 100.0
        
        if 'profile_completeness' in normalized and normalized['profile_completeness'] > 1:
            normalized['profile_completeness'] /= 100.0
    
    elif model_type == "fake_profile_detection":
        if 'profile_completeness' in normalized and normalized['profile_completeness'] > 1:
            normalized['profile_completeness'] /= 100.0
    
    return normalized


# ✅ Health route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'message': 'ML API is running!',
        'available_models': ['success_prediction', 'fake_profile_detection']
    })


# ✅ NEW: Root route (fix 404)
@app.route('/')
def home():
    return "ML API is running 🚀. Use /health or /api/ml/predict"


# ✅ Main ML API
@app.route('/api/ml/predict', methods=['POST'])
def ml_predict():
    try:
        data = request.json
        
        if not data or 'model' not in data or 'parameters' not in data:
            return jsonify({
                'success': False,
                'error': 'Invalid request format. Required: model, parameters'
            }), 400
        
        model_type = data['model']
        parameters = data['parameters']
        
        print(f"\n📥 Request: {model_type}")
        print(f"Parameters: {parameters}")
        
        if model_type == "success_prediction":
            normalized_params = normalize_parameters(parameters, "success_prediction")
            
            probability = success_predictor.predict_success_probability(normalized_params)
            
            if probability >= 75:
                recommendation = "High chance of success"
                risk_level = "Low"
            elif probability >= 50:
                recommendation = "Moderate chance of success"
                risk_level = "Medium"
            else:
                recommendation = "Low chance of success"
                risk_level = "High"
            
            return jsonify({
                'success': True,
                'model': 'success_prediction',
                'prediction': {
                    'success_probability': probability,
                    'recommendation': recommendation,
                    'risk_level': risk_level
                },
                'input_parameters': normalized_params
            })
        
        elif model_type == "fake_profile_detection":
            normalized_params = normalize_parameters(parameters, "fake_profile_detection")
            
            result = fake_detector.detect_fake(normalized_params)
            
            return jsonify({
                'success': True,
                'model': 'fake_profile_detection',
                'prediction': {
                    'is_fake': result['is_fake'],
                    'fake_probability': result['fake_probability'],
                    'risk_level': result['risk_level'],
                    'red_flags': result['red_flags']
                },
                'input_parameters': normalized_params
            })
        
        else:
            return jsonify({
                'success': False,
                'error': 'Unknown model type'
            }), 400
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ✅ Legacy endpoints
@app.route('/api/predict-success', methods=['POST'])
def predict_success():
    try:
        data = request.json
        normalized = normalize_parameters(data, "success_prediction")
        
        probability = success_predictor.predict_success_probability(normalized)
        
        if probability >= 75:
            recommendation = "High chance of success"
            emoji = "✅"
        elif probability >= 50:
            recommendation = "Moderate chance of success"
            emoji = "⚠️"
        else:
            recommendation = "Low chance of success"
            emoji = "❌"
        
        return jsonify({
            'success': True,
            'success_probability': probability,
            'recommendation': recommendation,
            'emoji': emoji
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/detect-fake', methods=['POST'])
def detect_fake():
    try:
        data = request.json
        normalized = normalize_parameters(data, "fake_profile_detection")
        
        result = fake_detector.detect_fake(normalized)
        
        return jsonify({
            'success': True,
            **result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


# ✅ MAIN
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting ML API Server (AI Gateway Compatible)")
    print("="*60)
    print("API running at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /")
    print("  GET  /health")
    print("  POST /api/ml/predict")
    print("  POST /api/predict-success")
    print("  POST /api/detect-fake")
    print("\nPress CTRL+C to stop")
    print("="*60 + "\n")
    
   
   
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))