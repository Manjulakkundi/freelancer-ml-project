import pandas as pd
import numpy as np

def generate_fake_profile_data(n_samples=1000):
    """
    Create dataset with both genuine and fake profiles
    """
    
    genuine_count = n_samples // 2  # Half genuine
    fake_count = n_samples - genuine_count  # Half fake
    
    print(f"Creating {genuine_count} genuine profiles...")
    print(f"Creating {fake_count} fake profiles...")
    
    # === GENUINE PROFILES (normal, realistic profiles) ===
    genuine_data = {
        'profile_completeness': np.random.uniform(0.6, 1.0, genuine_count),
        'total_skills': np.random.randint(3, 15, genuine_count),
        'avg_rating': np.random.uniform(3.5, 5.0, genuine_count),
        'total_reviews': np.random.randint(5, 100, genuine_count),
        'total_projects': np.random.randint(1, 80, genuine_count),
        'account_age_days': np.random.randint(90, 1825, genuine_count),
        'portfolio_items': np.random.randint(1, 20, genuine_count),
        'budget_ratio': np.random.uniform(0.7, 1.2, genuine_count),
        'has_certifications': np.random.choice([0, 1], genuine_count, p=[0.3, 0.7]),
        'is_fake': 0
    }
    
    # === FAKE PROFILES (suspicious, unrealistic profiles) ===
    fake_data = {
        'profile_completeness': np.random.uniform(0.1, 0.5, fake_count),
        'total_skills': np.random.randint(20, 50, fake_count),
        'avg_rating': np.random.choice([0, 5.0], fake_count, p=[0.3, 0.7]),
        'total_reviews': np.random.randint(0, 3, fake_count),
        'total_projects': np.random.randint(0, 5, fake_count),
        'account_age_days': np.random.randint(1, 60, fake_count),
        'portfolio_items': np.random.randint(0, 2, fake_count),
        'budget_ratio': np.random.uniform(0.1, 0.5, fake_count),
        'has_certifications': np.random.choice([0, 1], fake_count, p=[0.9, 0.1]),
        'is_fake': 1
    }
    
    # Combine both types
    df_genuine = pd.DataFrame(genuine_data)
    df_fake = pd.DataFrame(fake_data)
    df = pd.concat([df_genuine, df_fake], ignore_index=True)
    
    # Shuffle them together
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

# Generate the dataset
df_fake = generate_fake_profile_data(1000)
df_fake.to_csv('fake_profile_data.csv', index=False)

print("\n✅ Fake profile dataset created!")
print(f"Genuine profiles: {(df_fake['is_fake'] == 0).sum()}")
print(f"Fake profiles: {(df_fake['is_fake'] == 1).sum()}")