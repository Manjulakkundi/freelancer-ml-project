import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

class FreelancerSuccessPredictor:
    """
    This class is like a 'brain' that learns to predict success
    """
    
    def __init__(self):
        # Scaler: Makes all numbers comparable (like converting kg and grams to same unit)
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
    
    def train(self, X, y):
        """
        Train the model to learn patterns
        
        X = features (experience, ratings, etc.)
        y = target (success or failure)
        """
        
        print("🎓 Starting training...\n")
        
        # Step 1: Split data into training (80%) and testing (20%)
        # Why? We train on 80%, then test on 20% to see if it really learned!
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training on {len(X_train)} examples")
        print(f"Testing on {len(X_test)} examples\n")
        
        # Step 2: Scale the features
        # This makes sure 'experience_years' (0-15) and 'avg_rating' (0-5) 
        # are on similar scales
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Step 3: Try different ML algorithms to see which works best
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        best_score = 0
        best_model_name = None
        
        # Train each model and pick the best one
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Teach the model using training data
            model.fit(X_train_scaled, y_train)
            
            # Test it on data it hasn't seen
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # AUC-ROC score: measures how good the model is (0.5 = random, 1.0 = perfect)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            print(f"  AUC Score: {auc_score:.4f}")
            print(classification_report(y_test, y_pred, target_names=['Fail', 'Success']))
            print()
            
            # Keep track of the best model
            if auc_score > best_score:
                best_score = auc_score
                best_model_name = name
                self.model = model
        
        print(f"🏆 Best Model: {best_model_name} (AUC: {best_score:.4f})\n")
        
        # Show which features are most important
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("📊 Top 5 Most Important Features:")
            print(importance.head())
        
        return self
    
    def predict_success_probability(self, freelancer_features):
        """
        Predict success probability for a new freelancer
        
        Returns: percentage (0-100)
        """
        if isinstance(freelancer_features, dict):
            freelancer_features = pd.DataFrame([freelancer_features])
        
        # Scale the features the same way we did during training
        X_scaled = self.scaler.transform(freelancer_features)
        
        # Get probability of success (second column = probability of class 1)
        success_prob = self.model.predict_proba(X_scaled)[:, 1][0]
        
        return round(success_prob * 100, 2)
    
    def save_model(self, filepath='success_predictor.pkl'):
        """Save the trained model so we can use it later"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath='success_predictor.pkl'):
        """Load a previously saved model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        print(f"📂 Model loaded from {filepath}")
        return self


# === MAIN TRAINING SCRIPT ===
if __name__ == "__main__":
    # Load the data we created earlier
    df = pd.read_csv('freelancer_success_data.csv')
    
    # Separate features (X) from target (y)
    X = df.drop('success', axis=1)  # Everything except 'success'
    y = df['success']                # Just the 'success' column
    
    # Create and train the predictor
    predictor = FreelancerSuccessPredictor()
    predictor.feature_names = X.columns.tolist()
    predictor.train(X, y)
    
    # Save it for later use
    predictor.save_model()
    
    print("\n" + "="*50)
    print("🧪 TESTING WITH EXAMPLE FREELANCER")
    print("="*50)
    
    # Test with a sample freelancer
    test_freelancer = {
        'experience_years': 5.0,
        'total_projects': 45,
        'avg_rating': 4.7,
        'completion_rate': 0.95,
        'on_time_delivery_rate': 0.92,
        'skill_match_score': 0.85,
        'profile_completeness': 0.90,
        'budget_ratio': 0.95,
        
    }
    
    probability = predictor.predict_success_probability(test_freelancer)
    print(f"\n✨ Success Probability: {probability}%")
    
    if probability >= 75:
        print("✅ Recommendation: HIGH chance of success!")
    elif probability >= 50:
        print("⚠️  Recommendation: MODERATE chance of success")
    else:
        print("❌ Recommendation: LOW chance of success")