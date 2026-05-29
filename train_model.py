#!/usr/bin/env python3
"""
Model training script for disaster prediction system.
Loads collected data, trains Isolation Forest for anomaly detection,
and Random Forest for risk classification.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def load_data(csv_path):
    """Load data from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def preprocess_data(df):
    """Preprocess data: select features, handle missing values, scale."""
    # Define feature columns
    feature_cols = [
        # Weather features
        'weather_airPressure', 'weather_airQuality', 'weather_weather_risk_score', 'weather_weather_risk_level',
        # Marine features
        'marine_sea_level_anomaly', 'marine_swell_wave_height', 'marine_wave_height', 'marine_wave_period',
        'marine_wind_speed', 'marine_wind_wave_height',
        # News features
        'news_news_critical', 'news_news_high', 'news_news_medium', 'news_news_low',
        'news_news_total_articles', 'news_news_risk_score',
        # Geolocation features
        'geo_lat', 'geo_lon', 'geo_elevation', 'geo_distance_to_coast'
    ]
    
    # Extract features
    X = df[feature_cols].copy()
    
    # Handle missing values: fill with column mean
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, X_scaled, scaler, feature_cols

def train_isolation_forest(X_scaled):
    """Train Isolation Forest for anomaly detection."""
    # We'll use default parameters; contamination set to 0.1 assuming 10% anomalies
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X_scaled)
    return iso_forest

def assign_anomaly_labels(iso_forest, X_scaled):
    """Assign anomaly labels: -1 for anomaly, 1 for normal. Convert to 1 for anomaly, 0 for normal."""
    # Isolation Forest returns -1 for anomalies and 1 for normal
    y_pred = iso_forest.predict(X_scaled)
    # Convert to 1 for anomaly, 0 for normal
    y = np.where(y_pred == -1, 1, 0)
    return y

def get_real_disaster_labels(df):
    """Get real disaster labels from data if available, otherwise return None."""
    if 'disaster_label' in df.columns:
        # disaster_label: 0 = normal, 1 = flood/storm/event
        return df['disaster_label'].values
    return None

def train_random_forest(X_scaled, y):
    """Train Random Forest classifier."""
    if y is None or len(np.unique(y)) < 2:
        print("Warning: No valid labels available. Using Isolation Forest labels for training.")
        return None
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf_clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return rf_clf

def save_models(iso_forest, rf_clf, scaler, feature_cols, model_dir):
    """Save trained models and scaler."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(iso_forest, os.path.join(model_dir, 'isolation_forest.pkl'))
    joblib.dump(rf_clf, os.path.join(model_dir, 'random_forest.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.pkl'))
    print(f"Models saved to {model_dir}")

def main():
    """Main function to run training."""
    csv_path = "/var/www/html/ml/data/disaster_data_labeled.csv"
    model_dir = "/var/www/html/ml/models"
    
    print("Loading data...")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows.")
    
    print("Preprocessing data...")
    X, X_scaled, scaler, feature_cols = preprocess_data(df)
    print(f"Features: {feature_cols}")
    
    print("Training Isolation Forest...")
    iso_forest = train_isolation_forest(X_scaled)
    
    # Try to get real disaster labels first
    y_real = get_real_disaster_labels(df)
    
    if y_real is not None:
        print("Using real disaster labels for Random Forest training.")
        y = y_real
        print(f"Disaster events: {np.sum(y)} out of {len(y)}")
    else:
        print("No real disaster labels found. Using Isolation Forest anomaly labels.")
        print("Warning: This is for initial setup only. Add real disaster labels for production.")
        y = assign_anomaly_labels(iso_forest, X_scaled)
        print(f"Anomalies detected: {np.sum(y)} out of {len(y)}")
    
    print("Training Random Forest classifier...")
    rf_clf = train_random_forest(X_scaled, y)
    
    if rf_clf is None:
        print("Random Forest training skipped. Saving only Isolation Forest.")
        save_models(iso_forest, None, scaler, feature_cols, model_dir)
    else:
        print("Saving models...")
        save_models(iso_forest, rf_clf, scaler, feature_cols, model_dir)
    
    print("Training completed.")

if __name__ == "__main__":
    main()
