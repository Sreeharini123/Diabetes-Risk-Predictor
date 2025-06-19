# train_diabetes_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pickle

# Load dataset
df = pd.read_csv("diabetes_dataset.csv")

# Replace 0s in important features with NaN
features_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[features_to_clean] = df[features_to_clean].replace(0, np.nan)

# Fill missing values with median
imputer = SimpleImputer(strategy='median')
df[features_to_clean] = imputer.fit_transform(df[features_to_clean])

# Define X and y
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model to a .pkl file
with open("diabetes_rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as diabetes_rf_model.pkl")
