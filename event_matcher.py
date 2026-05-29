#!/usr/bin/env python3
"""
Event matching engine to label training data with real disaster events from EM-DAT.
Matches training data timestamps and locations with disaster events.
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
from city_geolocation import CITY_GEOLOCATION
import numpy as np

DISASTERS_PATH = "/var/www/html/ml/data/libya_disasters.csv"
TRAINING_DATA_PATH = "/var/www/html/ml/data/disaster_data.csv"
OUTPUT_PATH = "/var/www/html/ml/data/disaster_data_labeled.csv"

def load_disasters():
    """Load processed Libya disasters."""
    df = pd.read_csv(DISASTERS_PATH)
    df['start_date'] = pd.to_datetime(df['start_date']).dt.tz_localize(None)
    print(f"Loaded {len(df)} disaster events.")
    return df

def load_training_data():
    """Load training data."""
    df = pd.read_csv(TRAINING_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    print(f"Loaded {len(df)} training data rows.")
    return df

def match_event(row, disasters_df, time_window_days=7):
    """
    Match a training data row with disaster events.
    
    Args:
        row: Training data row
        disasters_df: DataFrame of disaster events
        time_window_days: Days before/after disaster to consider as affected
    
    Returns:
        Tuple (disaster_label, disaster_type, severity) or (0, None, 0)
    """
    timestamp = row['timestamp']
    city = row['city']
    
    # Find disasters within time window for this city
    city_disasters = disasters_df[disasters_df['cities'].str.contains(city, na=False, case=False)]
    
    if city_disasters.empty:
        return 0, None, 0
    
    # Check if timestamp is within time window of any disaster
    for _, disaster in city_disasters.iterrows():
        disaster_date = disaster['start_date']
        time_diff = abs((timestamp - disaster_date).days)
        
        if time_diff <= time_window_days:
            # Match found
            disaster_type = disaster['disaster_type']
            severity = disaster['severity']
            return 1, disaster_type, severity
    
    return 0, None, 0

def label_training_data(training_df, disasters_df, time_window_days=7):
    """Label training data with disaster events."""
    print(f"Labeling training data with {time_window_days}-day window...")
    
    labels = []
    disaster_types = []
    severities = []
    
    for idx, row in training_df.iterrows():
        label, dtype, severity = match_event(row, disasters_df, time_window_days)
        labels.append(label)
        disaster_types.append(dtype)
        severities.append(severity)
        
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(training_df)} rows...")
    
    training_df['disaster_label'] = labels
    training_df['disaster_type'] = disaster_types
    training_df['disaster_severity'] = severities
    
    labeled_count = sum(labels)
    print(f"Labeled {labeled_count} rows as disaster events out of {len(training_df)} total.")
    
    return training_df

def add_synthetic_disasters(training_df, disasters_df):
    """
    Add synthetic disaster labels for recent data that may not be in EM-DAT yet.
    This is a temporary solution to ensure we have both positive and negative labels.
    """
    print("Adding synthetic labels for model training...")
    
    # Get disaster dates from EM-DAT
    disaster_dates = disasters_df['start_date'].dt.date.unique()
    
    # For rows with no label, check if they have high risk features
    # This is a heuristic to create some positive labels
    for idx, row in training_df.iterrows():
        if row['disaster_label'] == 0:
            # Check for high weather risk
            if row.get('weather_weather_risk_level', 0) >= 2:  # HIGH or CRITICAL
                training_df.at[idx, 'disaster_label'] = 1
                training_df.at[idx, 'disaster_type'] = 'Storm'
                training_df.at[idx, 'disaster_severity'] = 2
            # Check for high marine risk
            elif row.get('marine_wave_height', 0) > 2.0:
                training_df.at[idx, 'disaster_label'] = 1
                training_df.at[idx, 'disaster_type'] = 'Flood'
                training_df.at[idx, 'disaster_severity'] = 2
    
    synthetic_count = training_df['disaster_label'].sum() - disasters_df.shape[0]
    print(f"Added {synthetic_count} synthetic labels based on risk features.")
    
    return training_df

def balance_dataset(training_df):
    """
    Balance the dataset by undersampling the majority class (non-disaster).
    This helps with model training when disasters are rare.
    """
    disaster_rows = training_df[training_df['disaster_label'] == 1]
    non_disaster_rows = training_df[training_df['disaster_label'] == 0]
    
    print(f"Before balancing: {len(disaster_rows)} disaster, {len(non_disaster_rows)} non-disaster")
    
    if len(disaster_rows) == 0:
        print("Warning: No disaster labels found. Cannot balance.")
        return training_df
    
    # Sample non-disaster rows to match disaster rows (1:3 ratio)
    target_ratio = 3
    target_non_disaster = min(len(non_disaster_rows), len(disaster_rows) * target_ratio)
    
    balanced_non_disaster = non_disaster_rows.sample(n=target_non_disaster, random_state=42)
    balanced_df = pd.concat([disaster_rows, balanced_non_disaster])
    
    print(f"After balancing: {len(disaster_rows)} disaster, {len(balanced_non_disaster)} non-disaster")
    
    return balanced_df.sample(frac=1, random_state=42)  # Shuffle

def main():
    """Main labeling function."""
    # Load data
    disasters_df = load_disasters()
    training_df = load_training_data()
    
    # Label training data
    labeled_df = label_training_data(training_df, disasters_df, time_window_days=7)
    
    # Add synthetic labels for recent data
    labeled_df = add_synthetic_disasters(labeled_df, disasters_df)
    
    # Balance dataset
    balanced_df = balance_dataset(labeled_df)
    
    # Save labeled data
    balanced_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved labeled and balanced data to {OUTPUT_PATH}")
    
    # Print summary
    print("\n=== Label Summary ===")
    print(f"Total rows: {len(balanced_df)}")
    print(f"Disaster labels: {balanced_df['disaster_label'].sum()}")
    print(f"Non-disaster labels: {len(balanced_df) - balanced_df['disaster_label'].sum()}")
    print("\nBy Disaster Type:")
    print(balanced_df[balanced_df['disaster_label'] == 1]['disaster_type'].value_counts())
    print("\nBy Severity:")
    print(balanced_df[balanced_df['disaster_label'] == 1]['disaster_severity'].value_counts())

if __name__ == "__main__":
    main()
