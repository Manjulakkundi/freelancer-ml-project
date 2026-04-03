# 🚀 ML API Deployment & Integration Guide

## 📋 Overview

This guide explains how to deploy your ML API and integrate it with the AI Gateway at `https://freelancerhub-loadbalancer.vercel.app`

---

## 🏗️ Architecture

```
Frontend → AI Gateway → Your ML API → ML Models → Response
                ↓
          MongoDB (fetches user data)
```

**Flow:**
1. Frontend sends request to AI Gateway with `userId` or `freelancerId + projectId`
2. AI Gateway fetches data from MongoDB and calculates parameters
3. AI Gateway calls YOUR ML API with calculated parameters
4. Your ML API runs the model and returns prediction
5. AI Gateway returns result to frontend

---

## 📂 File Structure

```
freelancer-ml-project/
├── generate_data.py                  # Generate training data
├── train_success_model.py            # Train success model
├── generate_fake_profiles.py         # Generate fake profile data
├── train_fake_detector.py            # Train fake detector
├── api_gateway_integration.py        # NEW: Main API for AI Gateway
├── test_gateway_api.py               # NEW: Test script
├── freelancer_success_data.csv       # Training data
├── fake_profile_data.csv             # Training data
├── success_predictor.pkl             # Trained model
├── fake_detector.pkl                 # Trained model
└── requirements.txt                  # Python dependencies
```

---

## 🔧 Setup Instructions

### 1. Install Dependencies

```bash
pip install flask flask-cors pandas numpy scikit-learn joblib
```

### 2. Train Models (if not already done)

```bash
# Generate data
python generate_data.py
python generate_fake_profiles.py

# Train models
python train_success_model.py
python train_fake_detector.py
```

This creates:
- `success_predictor.pkl`
- `fake_detector.pkl`

### 3. Start ML API Server

```bash
python api_gateway_integration.py
```

You should see:
```
🔄 Loading ML models...
📂 Model loaded from success_predictor.pkl
📂 Model loaded from fake_detector.pkl
✅ Models loaded!

==========================================================
🚀 Starting ML API Server (AI Gateway Compatible)
==========================================================
API running at: http://localhost:5000
...
```

### 4. Test Locally

In a new terminal:
```bash
python test_gateway_api.py
```

---

## 🌐 API Endpoints

### Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "message": "ML API is running!",
  "available_models": ["success_prediction", "fake_profile_detection"]
}
```

---

### Unified ML Prediction (AI Gateway)
```
POST /api/ml/predict
Content-Type: application/json
```

**Request Format:**
```json
{
  "model": "success_prediction" | "fake_profile_detection",
  "parameters": {
    // See below for each model
  }
}
```

---

## 📊 Request/Response Formats

### Success Prediction

**Request:**
```json
{
  "model": "success_prediction",
  "parameters": {
    "experience_years": 0.1,
    "total_projects": 3,
    "avg_rating": 3.7,
    "completion_rate": 100,           // Percentage (0-100)
    "on_time_delivery_rate": 100,     // Percentage (0-100)
    "skill_match_score": 33.3,        // Percentage (0-100)
    "profile_completeness": 50,       // Percentage (0-100)
    "budget_ratio": 0.78
  }
}
```

**Response:**
```json
{
  "success": true,
  "model": "success_prediction",
  "prediction": {
    "success_probability": 87.5,
    "recommendation": "High chance of success",
    "risk_level": "Low"
  },
  "input_parameters": {
    "experience_years": 0.1,
    "total_projects": 3,
    "avg_rating": 3.7,
    "completion_rate": 1.0,           // Converted to decimal
    "on_time_delivery_rate": 1.0,
    "skill_match_score": 0.333,
    "profile_completeness": 0.5,
    "budget_ratio": 0.78
  }
}
```

---

### Fake Profile Detection

**Request:**
```json
{
  "model": "fake_profile_detection",
  "parameters": {
    "profile_completeness": 50,       // Percentage (0-100)
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
```

**Response:**
```json
{
  "success": true,
  "model": "fake_profile_detection",
  "prediction": {
    "is_fake": false,
    "fake_probability": 15.3,
    "risk_level": "Low",
    "red_flags": []
  },
  "input_parameters": {
    "profile_completeness": 0.5,      // Converted to decimal
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
```

---

## 🔄 Parameter Normalization

**Important:** The API automatically converts percentages to decimals!

AI Gateway sends:
- `completion_rate: 100` (meaning 100%)
- `profile_completeness: 50` (meaning 50%)

API converts to:
- `completion_rate: 1.0` (decimal)
- `profile_completeness: 0.5` (decimal)

This happens automatically in the `normalize_parameters()` function.

---

## 🚀 Deployment Options

### Option 1: Deploy on Render.com (Recommended - FREE)

1. **Create account** at https://render.com
2. **Create new Web Service**
3. **Connect GitHub repository**
4. **Configure:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python api_gateway_integration.py`
   - Environment: Python 3

5. **Environment Variables:**
   ```
   FLASK_ENV=production
   ```

6. **Deploy!** 🎉

You'll get a URL like: `https://your-app-name.onrender.com`

---

### Option 2: Deploy on Railway.app (FREE)

1. **Create account** at https://railway.app
2. **New Project → Deploy from GitHub**
3. **Select your repository**
4. **Add start command:**
   ```
   python api_gateway_integration.py
   ```
5. **Deploy!**

---

### Option 3: Deploy on Heroku (Paid)

1. Install Heroku CLI
2. Create `Procfile`:
   ```
   web: python api_gateway_integration.py
   ```
3. Deploy:
   ```bash
   heroku create your-ml-api
   git push heroku main
   ```

---

### Option 4: Local Deployment with ngrok (Testing)

```bash
# Start API locally
python api_gateway_integration.py

# In another terminal, expose with ngrok
ngrok http 5000
```

You'll get a public URL like: `https://abc123.ngrok.io`

---

## 📝 Requirements.txt

Create this file in your project:

```txt
Flask==3.0.0
flask-cors==4.0.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.2
joblib==1.3.2
gunicorn==21.2.0
```

---

## 🔗 Integration with AI Gateway

### Step 1: Share Your API URL

Once deployed, share your API URL with your teammate:
```
https://your-ml-api.onrender.com
```

### Step 2: AI Gateway Configuration

Your teammate needs to configure the AI Gateway to call your endpoint:

```javascript
// In AI Gateway code
const ML_API_URL = "https://your-ml-api.onrender.com/api/ml/predict";

async function callMLModel(model, parameters) {
  const response = await fetch(ML_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: model,
      parameters: parameters
    })
  });
  
  return await response.json();
}
```

---

## 🧪 Testing Integration

### Test from AI Gateway Format

```bash
curl -X POST https://your-ml-api.onrender.com/api/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "success_prediction",
    "parameters": {
      "experience_years": 5.0,
      "total_projects": 45,
      "avg_rating": 4.7,
      "completion_rate": 95,
      "on_time_delivery_rate": 92,
      "skill_match_score": 85,
      "profile_completeness": 90,
      "budget_ratio": 0.95
    }
  }'
```

---

## ⚠️ Important Notes

1. **API is stateless** - No session storage needed
2. **Models are loaded once** on startup
3. **Automatic parameter conversion** - Percentages → Decimals
4. **CORS enabled** - Frontend can call directly if needed
5. **Health check** - Use `/health` to verify deployment

---

## 🐛 Troubleshooting

### Issue: Models not loading
**Solution:** Make sure `.pkl` files are in the same directory as `api_gateway_integration.py`

### Issue: Port already in use
**Solution:** Change port in `api_gateway_integration.py`:
```python
app.run(debug=True, host='0.0.0.0', port=8000)  # Changed from 5000
```

### Issue: Predictions are wrong
**Solution:** Check parameter normalization - percentages should be 0-100, not 0-1

---

## 📞 Support

If you need help, contact:
- Your teammate (AI Gateway integration)
- Check logs in deployment platform
- Test locally first with `test_gateway_api.py`

---

## ✅ Deployment Checklist

- [ ] Models trained and `.pkl` files generated
- [ ] `requirements.txt` created
- [ ] API tested locally
- [ ] Deployment platform selected
- [ ] API deployed successfully
- [ ] Health check endpoint accessible
- [ ] Test prediction working
- [ ] URL shared with teammate
- [ ] AI Gateway configured to call your API
- [ ] End-to-end test completed

---

**Good luck with your deployment! 🚀**
