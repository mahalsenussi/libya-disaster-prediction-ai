#!/usr/bin/env python3
"""
Test script to verify the model prediction works.
"""

import requests
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime, timezone
from city_geolocation import get_city_geolocation

# Load models and scaler
model_dir = "/var/www/html/ml/models"
iso_forest = joblib.load(os.path.join(model_dir, 'isolation_forest.pkl'))
rf_clf = joblib.load(os.path.join(model_dir, 'random_forest.pkl'))
scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
feature_cols = joblib.load(os.path.join(model_dir, 'feature_cols.pkl'))

# API base URL for data collection (same as before)
BASE_URL = "http://localhost:5000"

def make_api_request(url, params=None, timeout=10):
    """Make an API request and return JSON response or None on error."""
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error requesting {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {url}: {e}")
        return None

def extract_weather_features(city_data):
    """Extract relevant features from weather data for a city."""
    if not city_data:
        return {}
    features = {
        'airPressure': city_data.get('airPressure', np.nan),
        'airQuality': city_data.get('airQuality', np.nan),
        'weather_risk_score': city_data.get('danger_expectations', {}).get('risk_score', np.nan),
    }
    risk_level = city_data.get('danger_expectations', {}).get('overall_risk_level', 'LOW')
    risk_level_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}
    features['weather_risk_level'] = risk_level_map.get(risk_level, 0)
    return features

def extract_marine_features(city_data):
    """Extract relevant features from marine data for a city."""
    if not city_data:
        return {}
    features = {
        'sea_level_anomaly': city_data.get('sea_level_anomaly', np.nan),
        'swell_wave_height': city_data.get('swell_wave_height', np.nan),
        'wave_height': city_data.get('wave_height', np.nan),
        'wave_period': city_data.get('wave_period', np.nan),
        'wind_speed': city_data.get('wind_speed', np.nan),
        'wind_wave_height': city_data.get('wind_wave_height', np.nan),
    }
    return features

def extract_news_features(news_data):
    """Extract relevant features from news danger assessment data with NLP enhancements."""
    if not news_data:
        return {
            'news_critical': 0,
            'news_high': 0,
            'news_medium': 0,
            'news_low': 0,
            'news_total_articles': 0,
            'news_risk_score': 0.0,
            'news_keyword_flood': 0,
            'news_keyword_storm': 0,
            'news_keyword_evacuation': 0,
            'news_alert_intensity': 0.0
        }
    danger_assessment = news_data.get('danger_assessment', {})
    danger_level_summary = danger_assessment.get('danger_level_summary', {})
    evaluations = danger_assessment.get('evaluations', [])
    
    features = {
        'news_critical': danger_level_summary.get('CRITICAL', 0),
        'news_high': danger_level_summary.get('HIGH', 0),
        'news_medium': danger_level_summary.get('MEDIUM', 0),
        'news_low': danger_level_summary.get('LOW', 0),
        'news_total_articles': len(evaluations),
    }
    
    total = features['news_total_articles']
    if total > 0:
        features['news_risk_score'] = (features['news_critical']*3 + features['news_high']*2 + features['news_medium']*1) / total
    else:
        features['news_risk_score'] = 0.0
    
    # Extract NLP features from evaluations
    keyword_flood = 0
    keyword_storm = 0
    keyword_evacuation = 0
    alert_intensity = 0.0
    
    for eval_item in evaluations:
        text = eval_item.get('text', '').lower()
        if 'flood' in text:
            keyword_flood += 1
            alert_intensity += 2
        if 'storm' in text:
            keyword_storm += 1
            alert_intensity += 2
        if 'evacuation' in text:
            keyword_evacuation += 1
            alert_intensity += 3
        if 'emergency' in text:
            alert_intensity += 2
        if 'warning' in text:
            alert_intensity += 1
    
    features['news_keyword_flood'] = keyword_flood
    features['news_keyword_storm'] = keyword_storm
    features['news_keyword_evacuation'] = keyword_evacuation
    features['news_alert_intensity'] = alert_intensity
    
    return features

def collect_city_data(city):
    """Collect data for a specific city and return a feature row."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Fetch news data (global for Libya) with a longer timeout
    news_data = make_api_request(f"{BASE_URL}/api/news/libya-danger-assessment", timeout=30)
    news_features = extract_news_features(news_data)
    
    # Fetch weather data for the city
    weather_url = f"{BASE_URL}/api/weather/{city}"
    weather_data = make_api_request(weather_url)
    weather_features = extract_weather_features(weather_data)
    
    # Fetch marine data for the city
    marine_url = f"{BASE_URL}/api/marine-data"
    marine_params = {'location': city}
    marine_data = make_api_request(marine_url, params=marine_params)
    marine_features = extract_marine_features(marine_data)
    
    # Get geolocation data for the city
    geo_data = get_city_geolocation(city)
    
    # Combine features
    row = {
        'timestamp': timestamp,
        'city': city,
    }
    for key, value in weather_features.items():
        row[f'weather_{key}'] = value
    for key, value in marine_features.items():
        row[f'marine_{key}'] = value
    for key, value in news_features.items():
        row[f'news_{key}'] = value
    # Add geolocation features
    row['geo_lat'] = geo_data['lat']
    row['geo_lon'] = geo_data['lon']
    row['geo_elevation'] = geo_data['elevation']
    row['geo_distance_to_coast'] = geo_data['distance_to_coast']
    
    return row

def predict_risk(city):
    """Predict risk for a given city."""
    try:
        # Collect data for the city
        row = collect_city_data(city)
        if not row:
            return {'error': 'Failed to collect data for city'}
        
        # Convert to DataFrame
        df = pd.DataFrame([row])
        
        # Extract features
        X = df[feature_cols].copy()
        X = X.fillna(X.mean())  # handle missing values
        X_scaled = scaler.transform(X)
        
        # Predict using Random Forest classifier
        prediction = rf_clf.predict(X_scaled)[0]
        probability = rf_clf.predict_proba(X_scaled)[0].tolist()
        
        # Map prediction to risk level
        risk_level_map = {0: 'LOW', 1: 'HIGH'}
        risk_level = risk_level_map.get(int(prediction), 'UNKNOWN')
        
        response = {
            'city': city,
            'timestamp': row['timestamp'],
            'risk_prediction': int(prediction),
            'risk_level': risk_level,
            'risk_probability': {
                'low': probability[0],
                'high': probability[1]
            }
        }
        
        return response
    
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    # Test for Tripoli
    result = predict_risk('Tripoli')
    print("Prediction result for Tripoli:")
    print(result)
