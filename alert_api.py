#!/usr/bin/env python3
"""
Alert API for disaster prediction system.
Loads trained models and provides risk predictions for cities.
"""

import requests
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime, timezone
from flask import Flask, jsonify, request
from city_geolocation import get_city_geolocation
from risk_engine_v2 import RiskEngineV2
from time_features import load_history, compute_realtime_features

app = Flask(__name__)

# Load models and scaler (V1 for anomaly detection fallback)
model_dir = "/var/www/html/ml/models"
iso_forest = joblib.load(os.path.join(model_dir, 'isolation_forest.pkl'))

# Initialize V2 risk engine
risk_engine = RiskEngineV2()

# Load historical data for time-series features
history_df = load_history()

# Alert log storage (in production, use a database)
alert_log = []

# API base URL for data collection (same as before)
BASE_URL = "http://localhost:5000"

# Cache for fallback values (in production, use Redis or database)
api_cache = {
    'weather': {},
    'marine': {},
    'news': None
}

def make_api_request(url, params=None, timeout=10, use_cache=True):
    """Make an API request and return JSON response or None on error.
    
    With fallback: if API fails, return cached value if available.
    """
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error requesting {url}: {e}")
        if use_cache:
            # Try to return cached value
            if 'weather' in url:
                city = url.split('/')[-1]
                if city in api_cache['weather']:
                    print(f"Using cached weather data for {city}")
                    return api_cache['weather'][city]
            elif 'marine' in url:
                location = params.get('location', 'unknown') if params else 'unknown'
                if location in api_cache['marine']:
                    print(f"Using cached marine data for {location}")
                    return api_cache['marine'][location]
            elif 'news' in url and api_cache['news']:
                print("Using cached news data")
                return api_cache['news']
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
    # Cache successful news response
    if news_data:
        api_cache['news'] = news_data
    
    # Fetch weather data for the city
    weather_url = f"{BASE_URL}/api/weather/{city}"
    weather_data = make_api_request(weather_url)
    weather_features = extract_weather_features(weather_data)
    # Cache successful weather response
    if weather_data:
        api_cache['weather'][city] = weather_data
    
    # Fetch marine data for the city
    marine_url = f"{BASE_URL}/api/marine-data"
    marine_params = {'location': city}
    marine_data = make_api_request(marine_url, params=marine_params)
    marine_features = extract_marine_features(marine_data)
    # Cache successful marine response
    if marine_data:
        api_cache['marine'][city] = marine_data
    
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

@app.route('/api/predict_risk/<city>', methods=['GET'])
def predict_risk(city):
    """Predict risk for a given city using V2 hybrid risk engine."""
    try:
        # Collect data for the city
        row = collect_city_data(city)
        if not row:
            return jsonify({'error': 'Failed to collect data for city'}), 500
        
        # Add time-series features using historical context
        row_with_time = compute_realtime_features(row, history_df)
        
        # Use V2 risk engine for prediction
        risk_result = risk_engine.predict(row_with_time)
        
        # Check if alert should be triggered
        should_alert = risk_engine.should_trigger_alert(risk_result)
        
        # Log alert if triggered
        if should_alert:
            alert_entry = {
                'city': city,
                'timestamp': row['timestamp'],
                'risk_score': risk_result['total_risk_score'],
                'risk_level': risk_result['risk_level'],
                'component_scores': risk_result['component_scores']
            }
            alert_log.append(alert_entry)
            print(f"ALERT TRIGGERED for {city}: {risk_result['risk_level']} risk (score: {risk_result['total_risk_score']:.2f})")
        
        response = {
            'city': city,
            'timestamp': row['timestamp'],
            'risk_score': risk_result['total_risk_score'],
            'risk_level': risk_result['risk_level'],
            'alert_triggered': should_alert,
            'component_scores': risk_result['component_scores'],
            'weights': risk_result['weights'],
            'features_used': row_with_time
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in predict_risk: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get recent alerts."""
    return jsonify({'alerts': alert_log[-100:]})  # Return last 100 alerts

@app.route('/api/alerts/clear', methods=['POST'])
def clear_alerts():
    """Clear alert log."""
    global alert_log
    alert_log = []
    return jsonify({'status': 'alerts cleared'})

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback on prediction accuracy for real-world learning."""
    try:
        data = request.get_json()
        
        city = data.get('city')
        event_happened = data.get('event_happened')
        severity = data.get('severity', 0)
        false_alarm = data.get('false_alarm', False)
        timestamp = data.get('timestamp', datetime.now(timezone.utc).isoformat())
        notes = data.get('notes', '')
        
        if not city or event_happened is None:
            return jsonify({'error': 'Missing required fields: city, event_happened'}), 400
        
        # Store feedback (in production, save to database)
        feedback_entry = {
            'city': city,
            'event_happened': event_happened,
            'severity': severity,
            'false_alarm': false_alarm,
            'timestamp': timestamp,
            'notes': notes,
            'received_at': datetime.now(timezone.utc).isoformat()
        }
        
        # For now, store in memory (in production, use database)
        if not hasattr(submit_feedback, 'feedback_log'):
            submit_feedback.feedback_log = []
        submit_feedback.feedback_log.append(feedback_entry)
        
        print(f"Feedback received for {city}: event_happened={event_happened}, severity={severity}, false_alarm={false_alarm}")
        
        return jsonify({
            'status': 'feedback recorded',
            'feedback': feedback_entry
        })
    
    except Exception as e:
        print(f"Error in submit_feedback: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['GET'])
def get_feedback():
    """Get all feedback entries."""
    if not hasattr(submit_feedback, 'feedback_log'):
        submit_feedback.feedback_log = []
    return jsonify({'feedback': submit_feedback.feedback_log[-100:]})

if __name__ == '__main__':
    # Run the Flask app on port 5001 to avoid conflict with the existing app on 5000
    app.run(host='0.0.0.0', port=5002, debug=True)
