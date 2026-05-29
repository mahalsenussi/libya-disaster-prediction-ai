#!/usr/bin/env python3
"""
Data collection script for disaster prediction system.
Collects weather, news, and marine data from APIs and stores in a CSV file.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import os
import json
from city_geolocation import get_city_geolocation

# API base URL (assuming the Flask app is running on localhost:5000)
BASE_URL = "http://localhost:5000"

# List of Libyan cities for weather and marine data
CITIES = [
    'Tripoli', 'Benghazi', 'Misrata', 'Al Khums', 'Zuwara',
    'Derna', 'Tobruk', 'Al Bayda', 'Sirte', 'Zawiya',
    'Ajdabiya', 'Sabha', 'Ghat', 'Ubari', 'Murzuq'
]

# News endpoint for danger assessment
NEWS_ENDPOINT = f"{BASE_URL}/api/news/libya-danger-assessment"

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
            cache_key = f"{url}_{str(params)}"
            # Simple cache lookup based on URL pattern
            if 'weather' in url:
                city = params.get('city', url.split('/')[-1]) if params else url.split('/')[-1]
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
    # Convert overall_risk_level to numeric
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
    
    # Compute weighted news risk score
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

def collect_data():
    """Collect data from all APIs and return a list of dictionaries (one per city)."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Fetch news data (global for Libya) with a longer timeout
    news_data = make_api_request(NEWS_ENDPOINT, timeout=30)
    news_features = extract_news_features(news_data)
    # Cache successful news response
    if news_data:
        api_cache['news'] = news_data
    
    # Initialize list to hold rows for each city
    rows = []
    
    for city in CITIES:
        # Fetch weather data for the city
        weather_url = f"{BASE_URL}/api/weather/{city}"
        weather_data = make_api_request(weather_url)
        weather_features = extract_weather_features(weather_data)
        # Cache successful weather response
        if weather_data:
            api_cache['weather'][city] = weather_data
        
        # Fetch marine data for the city (using city name as location)
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
        # Add weather features with prefix 'weather_'
        for key, value in weather_features.items():
            row[f'weather_{key}'] = value
        # Add marine features with prefix 'marine_'
        for key, value in marine_features.items():
            row[f'marine_{key}'] = value
        # Add news features (same for all cities at this timestamp)
        for key, value in news_features.items():
            row[f'news_{key}'] = value
        # Add geolocation features with prefix 'geo_'
        row['geo_lat'] = geo_data['lat']
        row['geo_lon'] = geo_data['lon']
        row['geo_elevation'] = geo_data['elevation']
        row['geo_distance_to_coast'] = geo_data['distance_to_coast']
        
        rows.append(row)
    
    return rows

def save_to_csv(rows, csv_path):
    """Save the list of rows to a CSV file, appending if file exists."""
    df_new = pd.DataFrame(rows)
    
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)
    
    print(f"Saved {len(rows)} rows to {csv_path}")

def main():
    """Main function to run data collection."""
    csv_path = "/var/www/html/ml/data/disaster_data.csv"
    print(f"Starting data collection at {datetime.now(timezone.utc).isoformat()}")
    
    rows = collect_data()
    if rows:
        save_to_csv(rows, csv_path)
        print("Data collection completed successfully.")
    else:
        print("No data collected due to errors.")

if __name__ == "__main__":
    main()
