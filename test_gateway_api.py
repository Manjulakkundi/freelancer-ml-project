import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health check"""
    print("="*60)
    print("🔍 Testing Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}\n")


def test_fake_detection_gateway_format():
    """Test fake detection with AI Gateway format"""
    print("="*60)
    print("🚨 Testing Fake Detection (AI Gateway Format)")
    print("="*60)
    
    # This is the exact format AI Gateway will send
    payload = {
        "model": "fake_profile_detection",
        "parameters": {
            "profile_completeness": 50,      # Percentage (0-100)
            "total_skills": 16,
            "avg_rating": 3.7,
            "total_reviews": 3,
            "total_projects": 3,
            "account_age_days": 30,
            "portfolio_items": 3,
            "budget_ratio": 0.89,
            "has_certifications": 0
        }
    }
    
    print(f"Request Payload:\n{json.dumps(payload, indent=2)}\n")
    
    response = requests.post(
        f"{BASE_URL}/api/ml/predict",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}\n")


def test_success_prediction_gateway_format():
    """Test success prediction with AI Gateway format"""
    print("="*60)
    print("🎯 Testing Success Prediction (AI Gateway Format)")
    print("="*60)
    
    # This is the exact format AI Gateway will send
    payload = {
        "model": "success_prediction",
        "parameters": {
            "experience_years": 0.1,
            "total_projects": 3,
            "avg_rating": 3.7,
            "completion_rate": 100,          # Percentage (0-100)
            "on_time_delivery_rate": 100,    # Percentage (0-100)
            "skill_match_score": 33.3,       # Percentage (0-100)
            "profile_completeness": 50,      # Percentage (0-100)
            "budget_ratio": 0.78
        }
    }
    
    print(f"Request Payload:\n{json.dumps(payload, indent=2)}\n")
    
    response = requests.post(
        f"{BASE_URL}/api/ml/predict",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}\n")


def test_suspicious_profile():
    """Test with a suspicious profile"""
    print("="*60)
    print("🚨 Testing Suspicious Profile Detection")
    print("="*60)
    
    payload = {
        "model": "fake_profile_detection",
        "parameters": {
            "profile_completeness": 20,      # Low completion
            "total_skills": 35,              # Too many skills
            "avg_rating": 5.0,               # Perfect rating
            "total_reviews": 1,              # Very few reviews
            "total_projects": 0,             # No projects
            "account_age_days": 15,          # New account
            "portfolio_items": 0,            # No portfolio
            "budget_ratio": 0.25,            # Suspiciously cheap
            "has_certifications": 0
        }
    }
    
    print(f"Request Payload:\n{json.dumps(payload, indent=2)}\n")
    
    response = requests.post(
        f"{BASE_URL}/api/ml/predict",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}\n")


def test_high_success_profile():
    """Test with a high success probability profile"""
    print("="*60)
    print("✅ Testing High Success Profile")
    print("="*60)
    
    payload = {
        "model": "success_prediction",
        "parameters": {
            "experience_years": 5.0,
            "total_projects": 45,
            "avg_rating": 4.7,
            "completion_rate": 95,           # 95%
            "on_time_delivery_rate": 92,     # 92%
            "skill_match_score": 85,         # 85%
            "profile_completeness": 90,      # 90%
            "budget_ratio": 0.95
        }
    }
    
    print(f"Request Payload:\n{json.dumps(payload, indent=2)}\n")
    
    response = requests.post(
        f"{BASE_URL}/api/ml/predict",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 ML API TESTING SUITE (AI Gateway Format)")
    print("="*60 + "\n")
    
    try:
        # Run all tests
        test_health()
        test_fake_detection_gateway_format()
        test_success_prediction_gateway_format()
        test_suspicious_profile()
        test_high_success_profile()
        
        print("="*60)
        print("✅ All Tests Completed!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API server")
        print("Make sure the API is running: python api_gateway_integration.py")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
