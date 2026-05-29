#!/usr/bin/env python3
"""
Create synthetic training dataset based on EM-DAT disaster patterns.
Generates realistic disaster scenarios using current data as baseline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from city_geolocation import CITY_GEOLOCATION
import random

def add_noise(val, scale=0.4):
    """Add random noise to a value to prevent overfitting."""
    if pd.isna(val):
        return val
    noise = random.uniform(-scale, scale)
    return val * (1 + noise)

DISASTERS_PATH = "/var/www/html/ml/data/libya_disasters.csv"
BASE_DATA_PATH = "/var/www/html/ml/data/disaster_data.csv"
OUTPUT_PATH = "/var/www/html/ml/data/disaster_data_labeled.csv"

def load_base_data():
    """Load current data as baseline for synthetic generation."""
    df = pd.read_csv(BASE_DATA_PATH)
    print(f"Loaded {len(df)} baseline data rows.")
    return df

def load_disasters():
    """Load processed Libya disasters."""
    df = pd.read_csv(DISASTERS_PATH)
    df['start_date'] = pd.to_datetime(df['start_date'])
    print(f"Loaded {len(df)} disaster events.")
    return df

def generate_disaster_scenario(base_row, disaster_info):
    """
    Generate a synthetic disaster scenario based on a baseline row and disaster info.
    Modifies features to reflect disaster conditions.
    """
    row = base_row.copy()
    disaster_type = disaster_info['disaster_type']
    severity = disaster_info['severity']
    
    # Add geolocation features
    city = disaster_info['cities'][0] if disaster_info['cities'] else row['city']
    geo_data = CITY_GEOLOCATION.get(city, CITY_GEOLOCATION['Tripoli'])
    row['geo_lat'] = geo_data['lat']
    row['geo_lon'] = geo_data['lon']
    row['geo_elevation'] = geo_data['elevation']
    row['geo_distance_to_coast'] = geo_data['distance_to_coast']
    
    # Modify features based on disaster type and severity
    severity_multiplier = severity  # 1, 2, or 3
    
    # Add class overlap: sometimes disasters look mild (20% chance)
    if random.random() < 0.2:
        row['weather_weather_risk_level'] = random.choice([0, 1])
        row['weather_weather_risk_score'] = add_noise(random.uniform(0.1, 0.4))
    
    if disaster_type in ['Flood', 'Water']:
        # Increase wave height, sea level anomaly
        row['marine_wave_height'] = add_noise(min(row['marine_wave_height'] * (1 + 0.5 * severity_multiplier), 5.0))
        row['marine_sea_level_anomaly'] = add_noise(min(row['marine_sea_level_anomaly'] + 0.2 * severity_multiplier, 1.0))
        row['marine_wind_speed'] = add_noise(min(row['marine_wind_speed'] * (1 + 0.3 * severity_multiplier), 20.0))
        
        # Increase weather risk
        row['weather_weather_risk_level'] = min(row['weather_weather_risk_level'] + severity_multiplier, 3)
        row['weather_weather_risk_score'] = add_noise(min(row['weather_weather_risk_score'] + 0.2 * severity_multiplier, 1.0))
        
    elif disaster_type == 'Storm':
        # Increase wind speed, wave height
        row['marine_wind_speed'] = add_noise(min(row['marine_wind_speed'] * (1 + 0.8 * severity_multiplier), 25.0))
        row['marine_wave_height'] = add_noise(min(row['marine_wave_height'] * (1 + 0.6 * severity_multiplier), 6.0))
        row['marine_swell_wave_height'] = add_noise(min(row['marine_swell_wave_height'] * (1 + 0.5 * severity_multiplier), 4.0))
        
        # Increase weather risk
        row['weather_weather_risk_level'] = min(row['weather_weather_risk_level'] + severity_multiplier, 3)
        row['weather_airPressure'] = add_noise(max(row['weather_airPressure'] - 10 * severity_multiplier, 980))
        
    elif disaster_type == 'Earthquake':
        # Earthquakes don't directly affect weather/marine, but can trigger floods
        # Add news intensity
        row['news_news_critical'] = add_noise(row['news_news_critical'] + severity_multiplier)
        row['news_alert_intensity'] = add_noise(row['news_alert_intensity'] + 3 * severity_multiplier)
    
    # Add news features for all disasters
    row['news_news_critical'] = add_noise(min(row['news_news_critical'] + severity_multiplier, 10))
    row['news_alert_intensity'] = add_noise(min(row['news_alert_intensity'] + 2 * severity_multiplier, 20))
    
    # Add disaster keywords
    if disaster_type == 'Flood':
        row['news_keyword_flood'] = add_noise(row.get('news_keyword_flood', 0) + severity_multiplier)
    elif disaster_type == 'Storm':
        row['news_keyword_storm'] = add_noise(row.get('news_keyword_storm', 0) + severity_multiplier)
    
    # Update timestamp to disaster date
    row['timestamp'] = disaster_info['start_date']
    row['city'] = disaster_info['cities'][0] if disaster_info['cities'] else row['city']
    
    # Add labels
    row['disaster_label'] = 1
    row['disaster_type'] = disaster_type
    row['disaster_severity'] = severity
    
    # Add soft labels for borderline cases (15% chance)
    if random.random() < 0.15:
        row['disaster_label'] = random.choice([0, 1])
    
    # Add random feature dropout (simulate sensor failures - 15% chance)
    if random.random() < 0.15:
        if random.random() < 0.5:
            row['marine_wave_height'] = np.nan
        if random.random() < 0.5:
            row['marine_wind_speed'] = np.nan
        if random.random() < 0.5:
            row['weather_airPressure'] = np.nan
    
    return row

def generate_normal_scenario(base_row):
    """Generate a normal (non-disaster) scenario with low risk features."""
    row = base_row.copy()
    
    # Add geolocation features
    city = row['city']
    geo_data = CITY_GEOLOCATION.get(city, CITY_GEOLOCATION['Tripoli'])
    row['geo_lat'] = geo_data['lat']
    row['geo_lon'] = geo_data['lon']
    row['geo_elevation'] = geo_data['elevation']
    row['geo_distance_to_coast'] = geo_data['distance_to_coast']
    
    # Ensure low risk values with noise
    row['marine_wave_height'] = add_noise(min(row['marine_wave_height'], 1.0))
    row['marine_wind_speed'] = add_noise(min(row['marine_wind_speed'], 8.0))
    row['marine_sea_level_anomaly'] = add_noise(min(abs(row['marine_sea_level_anomaly']), 0.1))
    
    # Add class overlap: sometimes normal looks dangerous (10% chance)
    if random.random() < 0.1:
        row['weather_weather_risk_level'] = random.choice([2, 3])
        row['weather_weather_risk_score'] = add_noise(random.uniform(0.5, 0.8))
    else:
        row['weather_weather_risk_level'] = 0
        row['weather_weather_risk_score'] = add_noise(min(row['weather_weather_risk_score'], 0.2))
    
    row['news_news_critical'] = 0
    row['news_alert_intensity'] = 0
    row['news_keyword_flood'] = 0
    row['news_keyword_storm'] = 0
    row['news_keyword_evacuation'] = 0
    
    # Add labels
    row['disaster_label'] = 0
    row['disaster_type'] = None
    row['disaster_severity'] = 0
    
    # Add random feature dropout (simulate sensor failures - 15% chance)
    if random.random() < 0.15:
        if random.random() < 0.5:
            row['marine_wave_height'] = np.nan
        if random.random() < 0.5:
            row['marine_wind_speed'] = np.nan
        if random.random() < 0.5:
            row['weather_airPressure'] = np.nan
    
    return row

def generate_hard_disaster_scenario(base_row, disaster_info):
    """Generate a disaster scenario with low feature values (hard example)."""
    row = base_row.copy()
    disaster_type = disaster_info['disaster_type']
    severity = disaster_info['severity']
    
    # Add geolocation features
    city = disaster_info['cities'][0] if disaster_info['cities'] else row['city']
    geo_data = CITY_GEOLOCATION.get(city, CITY_GEOLOCATION['Tripoli'])
    row['geo_lat'] = geo_data['lat']
    row['geo_lon'] = geo_data['lon']
    row['geo_elevation'] = geo_data['elevation']
    row['geo_distance_to_coast'] = geo_data['distance_to_coast']
    
    # Keep features LOW but still mark as disaster (hard example)
    row['marine_wave_height'] = add_noise(min(row['marine_wave_height'], 1.5))
    row['marine_wind_speed'] = add_noise(min(row['marine_wind_speed'], 12.0))
    row['marine_sea_level_anomaly'] = add_noise(min(abs(row['marine_sea_level_anomaly']), 0.2))
    
    row['weather_weather_risk_level'] = 1  # MEDIUM instead of HIGH
    row['weather_weather_risk_score'] = add_noise(min(row['weather_weather_risk_score'], 0.4))
    
    # Low news intensity but still disaster
    row['news_news_critical'] = 0
    row['news_alert_intensity'] = add_noise(min(row['news_alert_intensity'], 3.0))
    row['news_keyword_flood'] = 0
    row['news_keyword_storm'] = 0
    
    # Update timestamp
    row['timestamp'] = disaster_info['start_date']
    row['city'] = city
    
    # Add labels
    row['disaster_label'] = 1
    row['disaster_type'] = disaster_type
    row['disaster_severity'] = 1  # Minor severity for hard examples
    
    return row

def generate_hard_normal_scenario(base_row):
    """Generate a normal scenario with high feature values (hard example)."""
    row = base_row.copy()
    
    # Add geolocation features
    city = row['city']
    geo_data = CITY_GEOLOCATION.get(city, CITY_GEOLOCATION['Tripoli'])
    row['geo_lat'] = geo_data['lat']
    row['geo_lon'] = geo_data['lon']
    row['geo_elevation'] = geo_data['elevation']
    row['geo_distance_to_coast'] = geo_data['distance_to_coast']
    
    # High features but NO disaster (hard example)
    row['marine_wave_height'] = add_noise(min(row['marine_wave_height'] * 1.5, 3.0))
    row['marine_wind_speed'] = add_noise(min(row['marine_wind_speed'] * 1.5, 15.0))
    row['marine_sea_level_anomaly'] = add_noise(min(abs(row['marine_sea_level_anomaly']) + 0.3, 0.5))
    
    row['weather_weather_risk_level'] = 2  # HIGH but no disaster
    row['weather_weather_risk_score'] = add_noise(min(row['weather_weather_risk_score'] + 0.3, 0.7))
    
    # High news but no disaster
    row['news_news_critical'] = add_noise(random.randint(1, 3))
    row['news_alert_intensity'] = add_noise(random.uniform(5, 10))
    row['news_keyword_flood'] = add_noise(random.randint(1, 3))
    row['news_keyword_storm'] = add_noise(random.randint(1, 2))
    
    # Randomize timestamp
    random_days = random.randint(0, 365)
    row['timestamp'] = datetime.now() - timedelta(days=random_days)
    
    # Add labels
    row['disaster_label'] = 0
    row['disaster_type'] = None
    row['disaster_severity'] = 0
    
    return row

def generate_confusing_scenario(base_row):
    """Generate confusing scenarios that blur class boundaries."""
    row = base_row.copy()
    
    # Add geolocation features
    city = row['city']
    geo_data = CITY_GEOLOCATION.get(city, CITY_GEOLOCATION['Tripoli'])
    row['geo_lat'] = geo_data['lat']
    row['geo_lon'] = geo_data['lon']
    row['geo_elevation'] = geo_data['elevation']
    row['geo_distance_to_coast'] = geo_data['distance_to_coast']
    
    # Randomly choose confusing scenario type
    scenario_type = random.choice(['high_wave_no_disaster', 'low_signal_disaster'])
    
    if scenario_type == 'high_wave_no_disaster':
        # High wave but no disaster (false positive scenario)
        row['marine_wave_height'] = add_noise(random.uniform(2.5, 4.0))
        row['marine_wind_speed'] = add_noise(random.uniform(12, 18))
        row['marine_sea_level_anomaly'] = add_noise(random.uniform(0.3, 0.6))
        row['weather_weather_risk_level'] = random.choice([1, 2])
        row['weather_weather_risk_score'] = add_noise(random.uniform(0.4, 0.7))
        row['news_alert_intensity'] = add_noise(random.uniform(3, 8))
        row['disaster_label'] = 0
        row['disaster_type'] = None
        row['disaster_severity'] = 0
        
    else:  # low_signal_disaster
        # Low signal but disaster happening (false negative scenario)
        row['marine_wave_height'] = add_noise(random.uniform(0.8, 1.5))
        row['marine_wind_speed'] = add_noise(random.uniform(8, 12))
        row['marine_sea_level_anomaly'] = add_noise(random.uniform(0.0, 0.2))
        row['weather_weather_risk_level'] = random.choice([0, 1])
        row['weather_weather_risk_score'] = add_noise(random.uniform(0.1, 0.4))
        row['news_alert_intensity'] = add_noise(random.uniform(0, 3))
        row['disaster_label'] = 1
        row['disaster_type'] = random.choice(['Flood', 'Water', 'Storm'])
        row['disaster_severity'] = 1
    
    # Randomize timestamp
    random_days = random.randint(0, 365)
    row['timestamp'] = datetime.now() - timedelta(days=random_days)
    
    # Add random feature dropout
    if random.random() < 0.15:
        if random.random() < 0.5:
            row['marine_wave_height'] = np.nan
        if random.random() < 0.5:
            row['marine_wind_speed'] = np.nan
    
    return row

def create_synthetic_dataset(base_df, disasters_df, samples_per_disaster=100):
    """Create synthetic training dataset with increased size and noise."""
    print("Creating synthetic training dataset...")
    
    synthetic_rows = []
    
    # Generate disaster scenarios
    for _, disaster in disasters_df.iterrows():
        # Use city from disaster if available, otherwise use random city
        if disaster['cities'] and pd.notna(disaster['cities']):
            cities = disaster['cities'].split(',')
            cities = [c.strip() for c in cities if c.strip() in CITY_GEOLOCATION]
        else:
            # Use random cities for disasters without location data
            cities = list(CITY_GEOLOCATION.keys())
        
        if not cities:
            cities = ['Tripoli']  # Fallback
        
        for city in cities[:2]:  # Limit to 2 cities per disaster to avoid explosion
            
            # Find base data for this city
            city_base = base_df[base_df['city'] == city]
            if city_base.empty:
                # Use any base data
                city_base = base_df.sample(1)
            
            base_row = city_base.iloc[0]
            
            # Generate multiple samples per disaster (increased from 5 to 50)
            for _ in range(samples_per_disaster):
                disaster_info = {
                    'disaster_type': disaster['disaster_type'],
                    'severity': disaster['severity'],
                    'start_date': disaster['start_date'],
                    'cities': [city]
                }
                synthetic_row = generate_disaster_scenario(base_row, disaster_info)
                synthetic_rows.append(synthetic_row)
    
    print(f"Generated {len(synthetic_rows)} disaster scenarios.")
    
    # Generate hard disaster examples (disasters with low features)
    hard_disaster_count = len(synthetic_rows) // 4  # 25% of disasters as hard examples
    for _ in range(hard_disaster_count):
        base_row = base_df.sample(1).iloc[0]
        disaster = disasters_df.sample(1).iloc[0]
        cities_str = disaster['cities'] if pd.notna(disaster['cities']) else 'Tripoli'
        city = cities_str.split(',')[0].strip() if isinstance(cities_str, str) else 'Tripoli'
        disaster_info = {
            'disaster_type': disaster['disaster_type'],
            'severity': 1,  # Always minor for hard examples
            'start_date': disaster['start_date'],
            'cities': [city]
        }
        hard_disaster_row = generate_hard_disaster_scenario(base_row, disaster_info)
        synthetic_rows.append(hard_disaster_row)
    
    print(f"Generated {hard_disaster_count} hard disaster examples.")
    
    # Generate normal scenarios (5x the disaster count for better balance)
    normal_count = len(synthetic_rows) * 5
    for _ in range(normal_count):
        base_row = base_df.sample(1).iloc[0]
        normal_row = generate_normal_scenario(base_row)
        # Randomize timestamp
        random_days = random.randint(0, 365)
        normal_row['timestamp'] = datetime.now() - timedelta(days=random_days)
        synthetic_rows.append(normal_row)
    
    print(f"Generated {normal_count} normal scenarios.")
    
    # Generate hard normal examples (normal with high features)
    hard_normal_count = normal_count // 10  # 10% of normals as hard examples
    for _ in range(hard_normal_count):
        base_row = base_df.sample(1).iloc[0]
        hard_normal_row = generate_hard_normal_scenario(base_row)
        synthetic_rows.append(hard_normal_row)
    
    print(f"Generated {hard_normal_count} hard normal examples.")
    
    # Generate confusing scenarios (blur class boundaries)
    confusing_count = len(synthetic_rows) // 20  # 5% confusing scenarios
    for _ in range(confusing_count):
        base_row = base_df.sample(1).iloc[0]
        confusing_row = generate_confusing_scenario(base_row)
        synthetic_rows.append(confusing_row)
    
    print(f"Generated {confusing_count} confusing scenarios.")
    print(f"Total synthetic rows: {len(synthetic_rows)}")
    
    return pd.DataFrame(synthetic_rows)

def main():
    """Main synthetic dataset generation."""
    # Load data
    base_df = load_base_data()
    disasters_df = load_disasters()
    
    # Ensure base data has all required columns
    required_cols = ['news_keyword_flood', 'news_keyword_storm', 'news_keyword_evacuation', 'news_alert_intensity']
    for col in required_cols:
        if col not in base_df.columns:
            base_df[col] = 0
    
    # Create synthetic dataset with reduced samples for realistic chaos
    synthetic_df = create_synthetic_dataset(base_df, disasters_df, samples_per_disaster=50)
    
    # Shuffle
    synthetic_df = synthetic_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    synthetic_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved synthetic dataset to {OUTPUT_PATH}")
    
    # Print summary
    print("\n=== Dataset Summary ===")
    print(f"Total rows: {len(synthetic_df)}")
    print(f"Disaster labels: {synthetic_df['disaster_label'].sum()}")
    print(f"Non-disaster labels: {len(synthetic_df) - synthetic_df['disaster_label'].sum()}")
    print("\nBy Disaster Type:")
    print(synthetic_df[synthetic_df['disaster_label'] == 1]['disaster_type'].value_counts())
    print("\nBy Severity:")
    print(synthetic_df[synthetic_df['disaster_label'] == 1]['disaster_severity'].value_counts())

if __name__ == "__main__":
    main()
