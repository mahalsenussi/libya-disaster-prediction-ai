#!/usr/bin/env python3
"""
V2 Model Training with XGBoost and Time-Series Features.
Upgrades from rule-based to learned AI model.
"""

import pandas as pd
import numpy as np
import joblib
import os

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

DATA_PATH = "/var/www/html/ml/data/disaster_data_labeled.csv"
MODEL_DIR = "/var/www/html/ml/models_v2"

def load_data():
    """Load labeled training data."""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")
    return df

def add_time_features(df):
    """Add time-series features for temporal intelligence."""
    df = df.sort_values(by=["city", "timestamp"])
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by city for time-series calculations
    df['wave_6h_avg'] = df.groupby('city')['marine_wave_height'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['wave_delta'] = df['marine_wave_height'] - df['wave_6h_avg']
    
    df['pressure_diff'] = df.groupby('city')['weather_airPressure'].diff().fillna(0)
    
    df['news_rolling'] = df.groupby('city')['news_news_total_articles'].transform(lambda x: x.rolling(24, min_periods=1).mean())
    df['news_spike'] = df['news_news_total_articles'] - df['news_rolling']
    
    return df

def preprocess(df):
    """Preprocess data with time-series features."""
    df = add_time_features(df)
    
    feature_cols = [
        # Weather features
        'weather_airPressure', 'weather_airQuality',
        'weather_weather_risk_score', 'weather_weather_risk_level',
        
        # Marine features
        'marine_wave_height', 'marine_wind_speed', 'marine_sea_level_anomaly',
        
        # News features
        'news_news_risk_score', 'news_alert_intensity',
        
        # Geolocation features
        'geo_lat', 'geo_lon', 'geo_elevation', 'geo_distance_to_coast',
        
        # NEW time-series features
        'wave_delta', 'pressure_diff', 'news_spike'
    ]
    
    X = df[feature_cols].copy().fillna(0)
    y = df['disaster_label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_cols

def train():
    """Train XGBoost model with real disaster labels."""
    df = load_data()
    X, y, scaler, feature_cols = preprocess(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Disaster ratio in train: {y_train.mean():.2%}")
    
    # Train XGBoost with adjusted parameters to use more features
    base_model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        eval_metric='logloss'
    )
    
    print("Training XGBoost model...")
    base_model.fit(X_train, y_train)
    
    # Calibrate probabilities to reduce overconfidence
    print("Calibrating probabilities...")
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method='isotonic',
        cv=3
    )
    calibrated_model.fit(X_train, y_train)
    
    model = calibrated_model
    
    # Evaluate
    preds = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # Probability histogram to check for overconfidence
    probs = model.predict_proba(X_test)[:, 1]
    print("\nProbability Distribution:")
    print(f"  Min: {probs.min():.4f}")
    print(f"  Max: {probs.max():.4f}")
    print(f"  Mean: {probs.mean():.4f}")
    print(f"  Std: {probs.std():.4f}")
    
    # Check for overconfidence
    extreme_probs = (probs < 0.1).sum() + (probs > 0.9).sum()
    print(f"  Extreme probabilities (<0.1 or >0.9): {extreme_probs}/{len(probs)} ({extreme_probs/len(probs):.1%})")
    
    # Feature importance (from base model, not calibrated)
    print("\nFeature Importance:")
    for feat, imp in sorted(zip(feature_cols, base_model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.4f}")
    
    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/xgb_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(feature_cols, f"{MODEL_DIR}/features.pkl")
    
    print(f"\n✅ V2 model saved to {MODEL_DIR}")

if __name__ == "__main__":
    train()
