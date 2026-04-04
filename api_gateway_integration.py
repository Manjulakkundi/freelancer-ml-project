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

# =========================
# REQUIRED FEATURES
# =========================

# These MUST match exactly what the model was trained on
SUCCESS_FEATURES = [
    'experience_years',
    'total_projects',
    'avg_rating',
    'completion_rate',
    'on_time_delivery_rate',
    'skill_match_score',
    'profile_completeness',
    'budget_ratio'
]

FAKE_FEATURES = [
    'profile_completeness',
    'total_skills',
    'avg_rating',
    'total_reviews',
    'total_projects',
    'account_age_days',
    'portfolio_items',
    'budget_ratio',
    'has_certifications'
]

# =========================
# LOAD MODELS
# =========================
print("🔄 Loading ML models...")
print("FILES IN DIRECTORY:", os.listdir())

try:
    success_predictor = FreelancerSuccessPredictor()
    success_model_path = os.path.join(os.getcwd(), 'success_predictor.pkl')
    success_predictor.load_model(success_model_path)

    fake_detector = FakeProfileDetector()
    fake_model_path = os.path.join(os.getcwd(), 'fake_detector.pkl')
    fake_detector.load_model(fake_model_path)

    print("✅ Models loaded successfully!\n")

except Exception as e:
    print("❌ Error loading models:", str(e))
    raise e


# =========================
# NORMALIZATION FUNCTION
# =========================
def normalize_parameters(params, model_type):
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
        if 'budget_ratio' in normalized and normalized['budget_ratio'] > 1:
            normalized['budget_ratio'] /= 100.0

    elif model_type == "fake_profile_detection":
        if 'profile_completeness' in normalized and normalized['profile_completeness'] > 1:
            normalized['profile_completeness'] /= 100.0
        if 'budget_ratio' in normalized and normalized['budget_ratio'] > 1:
            normalized['budget_ratio'] /= 100.0

    return normalized


# =========================
# VALIDATION FUNCTIONS
# =========================
def validate_success_features(params):
    missing = [f for f in SUCCESS_FEATURES if f not in params]
    return missing

def validate_fake_features(params):
    missing = [f for f in FAKE_FEATURES if f not in params]
    return missing


# =========================
# ROUTES
# =========================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'message': 'ML API is running!',
        'available_models': ['success_prediction', 'fake_profile_detection']
    })


@app.route('/')
def home():
    return "ML API is running 🚀. Use /health or /api/ml/predict"


@app.route('/api/ml/predict', methods=['POST'])
def ml_predict():
    try:
        data = request.json

        if not data or 'model' not in data or 'parameters' not in data:
            return jsonify({
                'success': False,
                'error': 'Invalid request format. Required fields: model, parameters'
            }), 400

        model_type = data['model']
        parameters = data['parameters']

        print(f"\n📥 Request: {model_type}")
        print(f"Parameters: {parameters}")

        if model_type == "success_prediction":

            missing = validate_success_features(parameters)
            if missing:
                return jsonify({
                    'success': False,
                    'error': f'Missing required fields: {missing}',
                    'required_fields': SUCCESS_FEATURES
                }), 400

            normalized_params = normalize_parameters(parameters, "success_prediction")
            ordered_params = {f: normalized_params[f] for f in SUCCESS_FEATURES}

            probability = success_predictor.predict_success_probability(ordered_params)

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
                'input_parameters': ordered_params
            })

        elif model_type == "fake_profile_detection":

            missing = validate_fake_features(parameters)
            if missing:
                return jsonify({
                    'success': False,
                    'error': f'Missing required fields: {missing}',
                    'required_fields': FAKE_FEATURES
                }), 400

            normalized_params = normalize_parameters(parameters, "fake_profile_detection")
            ordered_params = {f: normalized_params[f] for f in FAKE_FEATURES}

            result = fake_detector.detect_fake(ordered_params)

            return jsonify({
                'success': True,
                'model': 'fake_profile_detection',
                'prediction': {
                    'is_fake': result['is_fake'],
                    'fake_probability': result['fake_probability'],
                    'risk_level': result['risk_level'],
                    'red_flags': result['red_flags']
                },
                'input_parameters': ordered_params
            })

        else:
            return jsonify({
                'success': False,
                'error': f'Unknown model type: {model_type}. Use: success_prediction or fake_profile_detection'
            }), 400

    except Exception as e:
        import traceback
        print(f"❌ Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/predict-success', methods=['POST'])
def predict_success():
    try:
        data = request.json

        missing = validate_success_features(data)
        if missing:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {missing}',
                'required_fields': SUCCESS_FEATURES
            }), 400

        normalized = normalize_parameters(data, "success_prediction")
        ordered_params = {f: normalized[f] for f in SUCCESS_FEATURES}

        probability = success_predictor.predict_success_probability(ordered_params)

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
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/detect-fake', methods=['POST'])
def detect_fake():
    try:
        data = request.json

        missing = validate_fake_features(data)
        if missing:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {missing}',
                'required_fields': FAKE_FEATURES
            }), 400

        normalized = normalize_parameters(data, "fake_profile_detection")
        ordered_params = {f: normalized[f] for f in FAKE_FEATURES}

        result = fake_detector.detect_fake(ordered_params)

        return jsonify({
            'success': True,
            **result
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


# Shows exactly what fields are required for each model
@app.route('/api/fields', methods=['GET'])
def get_required_fields():
    return jsonify({
        'success_prediction': {
            'required_fields': SUCCESS_FEATURES,
            'example': {
                'experience_years': 5.0,
                'total_projects': 45,
                'avg_rating': 4.7,
                'completion_rate': 95,
                'on_time_delivery_rate': 92,
                'skill_match_score': 85,
                'profile_completeness': 90,
                'budget_ratio': 95
            }
        },
        'fake_profile_detection': {
            'required_fields': FAKE_FEATURES,
            'example': {
                'profile_completeness': 20,
                'total_skills': 35,
                'avg_rating': 5.0,
                'total_reviews': 1,
                'total_projects': 0,
                'account_age_days': 15,
                'portfolio_items': 0,
                'budget_ratio': 25,
                'has_certifications': 0
            }
        },
        'note': 'profile_completeness and budget_ratio can be 0-100 or 0-1, both are handled automatically'
    })


# =========================
# MAIN ENTRY (Render compatible)
# =========================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting ML API Server (Render Ready)")
    print("="*60)

    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))