"""
Script to train the battery RUL prediction model
Run this script to generate best_model.pkl and scaler.pkl
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("Loading dataset...")
df = pd.read_csv('Battery_dataset.csv')
print(f"Dataset shape: {df.shape}")

# Prepare data
print("\nPreparing data...")
data = df.copy()

# Encode battery_id
le = LabelEncoder()
data['battery_id'] = le.fit_transform(data['battery_id'])

# Split features and target
X = data.drop(columns=['RUL'])
y = data['RUL']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gradient Boosting model
print("\nTraining Gradient Boosting model...")
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42,
    verbose=1
)

# Train model
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Test with some sample predictions to verify model is working
print("\n" + "="*70)
print("Testing model with sample inputs:")
print("="*70)

test_samples = [
    [10, 1.45, 4.25, 26.0, 2.0, 3.6, 32.0, 1.95, 98.0, 0],  # Early cycle, good health
    [100, 1.40, 4.20, 27.0, 1.95, 3.5, 33.0, 1.50, 75.0, 0],  # Mid cycle
    [200, 1.25, 4.05, 29.0, 1.80, 3.3, 35.0, 1.00, 50.0, 0],  # Late cycle, poor health
]

for i, sample in enumerate(test_samples, 1):
    sample_scaled = scaler.transform([sample])
    pred = model.predict(sample_scaled)[0]
    print(f"Sample {i}: Cycle={sample[0]:3d}, SOH={sample[8]:5.1f}, BCt={sample[7]:.2f} -> Predicted RUL={pred:6.1f}")

# Save model and scaler
print("\n" + "="*70)
print("Saving model and scaler...")
joblib.dump(model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved successfully!")
print("You can now run the Streamlit app with: streamlit run app.py")
print("="*70)
