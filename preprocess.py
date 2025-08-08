import joblib
import os
import pandas as pd

def load_scaler_and_features(output_dir="output"):
    scaler = joblib.load(os.path.join(output_dir, "scaler.joblib"))
    with open(os.path.join(output_dir, "feature_list.txt")) as f:
        features = [line.strip() for line in f]
    return scaler, features

def preprocess_input(df, scaler, features):
    # Ensure all required features are present
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")
    # Select and order columns
    X = df[features]
    # Fill missing values with median (as in training)
    X = X.fillna(X.median())
    # Scale features
    X_scaled = scaler.transform(X)
    return X_scaled
