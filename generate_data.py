import pandas as pd
import numpy as np

def generate_training_data(n_samples=5000):
    """
    Create fake freelancer data for training
    Each row = one freelancer-project combination
    """
    
    # Random freelancer features
    data = {
        # How many years have they been working?
        'experience_years': np.random.uniform(0, 15, n_samples),
        
        # How many projects have they completed?
        'total_projects': np.random.randint(0, 100, n_samples),
        
        # Their average rating (out of 5 stars)
        'avg_rating': np.random.uniform(2.5, 5.0, n_samples),
        
        # What % of projects do they complete?
        'completion_rate': np.random.uniform(0.5, 1.0, n_samples),
        
        # What % do they deliver on time?
        'on_time_delivery_rate': np.random.uniform(0.4, 1.0, n_samples),
        
        # How well do their skills match this project? (0-1)
        'skill_match_score': np.random.uniform(0.3, 1.0, n_samples),
        
        # How complete is their profile? (0-1)
        'profile_completeness': np.random.uniform(0.3, 1.0, n_samples),
        
        # Their bid compared to budget (0.5 = half price, 1.5 = 50% over budget)
        'budget_ratio': np.random.uniform(0.5, 1.5, n_samples),
        
       
    }
    
    # Now create the TARGET: did they succeed or fail?
    # We calculate a "success score" based on their features
    success_score = (
        data['completion_rate'] * 0.25 +           # 25% weight
        data['on_time_delivery_rate'] * 0.20 +     # 20% weight
        data['skill_match_score'] * 0.20 +         # 20% weight
        (data['avg_rating'] / 5.0) * 0.15 +        # 15% weight
        data['profile_completeness'] * 0.10 +      # 10% weight
        np.clip(1 - data['budget_ratio'], 0, 1) * 0.10  # 10% weight
    )
    
    # Add some randomness (real life isn't perfect!)
    noise = np.random.normal(0, 0.1, n_samples)
    success_prob = np.clip(success_score + noise, 0, 1)
    
    # Convert to binary: 1 = success, 0 = failure
    # If success_prob > 0.6, we say they succeeded
    data['success'] = (success_prob > 0.6).astype(int)
    
    return pd.DataFrame(data)

# Generate 1000 fake freelancer records
df = generate_training_data(5000)

# Save to CSV file
df.to_csv('freelancer_success_data.csv', index=False)

print("✅ Created training data!")
print(f"Total records: {len(df)}")
print(f"Successful projects: {df['success'].sum()}")
print(f"Failed projects: {len(df) - df['success'].sum()}")