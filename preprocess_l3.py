import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ensure the models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# 1. Load Dataset 3
try:
    df = pd.read_csv('data/dataset_3.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: data/dataset_3.csv not found. Check your folder path.")
    exit()

# 2. Define the core Behavioral Columns (The "Signal")
L3_RAW_COLS = [
    'keyboard_input_speed', 'input_timing_consistency', 'input_pause_patterns',
    'app_switching_frequency', 'recognized_screen_sharing_apps',
    'pin_entry_speed', 'time_pressure_indicators', 
    'time_between_otp_generation_and_input', 'otp_request_frequency',
    'session_duration', 'authentication_attempts', 'handle_similarity_score'
]

# Create a copy with only the columns we need + the label
df_l3 = df[L3_RAW_COLS + ['is_fraud']].copy()

# 3. Numeric Conversion Fix (Prevents "str + int" errors)
cols_to_fix = ['app_switching_frequency', 'recognized_screen_sharing_apps', 'time_pressure_indicators']
for col in cols_to_fix:
    df_l3[col] = pd.to_numeric(df_l3[col], errors='coerce').fillna(0)

# 4. Feature Engineering: Scam Stress Index
df_l3['scam_stress_index'] = (df_l3['app_switching_frequency'] * df_l3['recognized_screen_sharing_apps']) + \
                             df_l3['time_pressure_indicators']

print("Successfully calculated scam_stress_index.")

# 5. Filter for 'Normal' behavior ONLY for training
# The Autoencoder must only see what "Good" looks like
train_data = df_l3[df_l3['is_fraud'] == 0].drop('is_fraud', axis=1)

# 6. Scaling (Z-score normalization)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)

# 7. Save the Scaler (Partner A will need this for the API)
joblib.dump(scaler, 'models/l3_scaler.pkl')

print(f"Preprocessing Complete.")
print(f"Final Feature Count: {train_scaled.shape[1]}")
print(f"Training Samples: {train_scaled.shape[0]}")
print("Files Saved: models/l3_scaler.pkl")

# We keep train_scaled in memory for the next script
# In a real flow, you'd pass this to your training function