import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

class FakeProfileDetector:
    """
    Detects if a freelancer profile is fake/suspicious
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
    
    def engineer_features(self, df):
        """
        Create new features that help detect fakes
        """
        df_eng = df.copy()
        
        # Derived features
        df_eng['projects_per_day'] = df['total_projects'] / (df['account_age_days'] + 1)
        df_eng['reviews_per_project'] = df['total_reviews'] / (df['total_projects'] + 1)
        df_eng['rating_review_ratio'] = df['avg_rating'] * df['total_reviews']
        df_eng['completeness_portfolio_ratio'] = df['profile_completeness'] / (df['portfolio_items'] + 1)
        
        # Red flag indicators
        df_eng['too_many_skills'] = (df['total_skills'] > 25).astype(int)
        df_eng['too_few_reviews'] = (df['total_reviews'] < 3).astype(int)
        df_eng['suspiciously_cheap'] = (df['budget_ratio'] < 0.4).astype(int)
        df_eng['new_account'] = (df['account_age_days'] < 30).astype(int)
        
        return df_eng
    
    def train(self, X, y):
        """Train the fake detector"""
        
        print("🎓 Starting fake detector training...\n")
        
        # Step 1: Engineer features
        X_eng = self.engineer_features(X)
        
        # Step 2: Split data (using stratify to maintain class balance)
        X_train, X_test, y_train, y_test = train_test_split(
            X_eng, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}\n")
        
        # Step 3: Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Step 4: Train Random Forest
        print("🌲 Training Random Forest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            class_weight='balanced'  # Handles class imbalance
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Step 5: Evaluate performance
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("\n📊 Model Performance:")
        print(classification_report(y_test, y_pred, target_names=['Genuine', 'Fake']))
        print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}\n")
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"  True Genuine: {cm[0][0]}, False Fake: {cm[0][1]}")
        print(f"  False Genuine: {cm[1][0]}, True Fake: {cm[1][1]}\n")
        
        # Show important features
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': X_eng.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("🔍 Top 5 Features for Detecting Fakes:")
            print(importance.head())
        
        return self
    
    def detect_fake(self, profile_features):
        """
        Detect if a profile is fake
        """
        if isinstance(profile_features, dict):
            profile_features = pd.DataFrame([profile_features])
        
        # Engineer features
        profile_eng = self.engineer_features(profile_features)
        
        # Scale
        X_scaled = self.scaler.transform(profile_eng)
        
        # Predict
        is_fake = self.model.predict(X_scaled)[0]
        fake_prob = self.model.predict_proba(X_scaled)[:, 1][0] * 100
        
        # Determine risk level
        if fake_prob < 30:
            risk_level = 'Low'
        elif fake_prob < 60:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        # Identify specific red flags
        red_flags = []
        
        if profile_features['profile_completeness'].iloc[0] < 0.4:
            red_flags.append("Incomplete profile")
        
        if profile_features['total_skills'].iloc[0] > 25:
            red_flags.append("Unrealistic number of skills")
        
        if profile_features['portfolio_items'].iloc[0] == 0:
            red_flags.append("No portfolio items")
        
        if (profile_features['total_reviews'].iloc[0] < 3 and 
            profile_features['avg_rating'].iloc[0] > 4.5):
            red_flags.append("High rating with very few reviews")
        
        if profile_features['account_age_days'].iloc[0] < 30:
            red_flags.append("Very new account")
        
        if profile_features['budget_ratio'].iloc[0] < 0.4:
            red_flags.append("Suspiciously low pricing")
        
        return {
            'is_fake': bool(is_fake),
            'fake_probability': round(fake_prob, 2),
            'risk_level': risk_level,
            'red_flags': red_flags
        }
    
    def save_model(self, filepath='fake_detector.pkl'):
        """Save the trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath='fake_detector.pkl'):
        """Load a previously saved model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        print(f"📂 Model loaded from {filepath}")
        return self


# === MAIN TRAINING SCRIPT ===
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('fake_profile_data.csv')
    
    # Separate features and target
    X = df.drop('is_fake', axis=1)
    y = df['is_fake']
    
    # Train detector
    detector = FakeProfileDetector()
    detector.feature_names = X.columns.tolist()
    detector.train(X, y)
    
    # Save model
    detector.save_model()
    
    print("\n" + "="*50)
    print("🧪 TESTING WITH SUSPICIOUS PROFILE")
    print("="*50)
    
    # Test with suspicious profile
    suspicious_profile = {
        'profile_completeness': 0.2,
        'total_skills': 35,
        'avg_rating': 5.0,
        'total_reviews': 1,
        'total_projects': 0,
        'account_age_days': 15,
        'portfolio_items': 0,
        'budget_ratio': 0.25,
        'has_certifications': 0
    }
    
    result = detector.detect_fake(suspicious_profile)
    
    print(f"\n🚨 Detection Result:")
    print(f"  Is Fake: {result['is_fake']}")
    print(f"  Fake Probability: {result['fake_probability']}%")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Red Flags Found: {len(result['red_flags'])}")
    for flag in result['red_flags']:
        print(f"    ⚠️  {flag}")