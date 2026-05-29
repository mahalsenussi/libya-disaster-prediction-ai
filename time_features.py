#!/usr/bin/env python3
"""
Time-series feature computation for real-time predictions.
Computes rolling averages, deltas, and spikes for temporal intelligence.
"""

import pandas as pd
import numpy as np

HISTORY_PATH = "/var/www/html/ml/data/disaster_data.csv"
HISTORY_SIZE = 500  # Number of historical rows to keep for context

def load_history():
    """Load historical data for time-series context."""
    try:
        df = pd.read_csv(HISTORY_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.tail(HISTORY_SIZE)
    except Exception as e:
        print(f"Warning: Could not load history: {e}")
        return pd.DataFrame()

def compute_realtime_features(current_row, history_df):
    """
    Compute time-series features for a single prediction.
    
    Args:
        current_row: Dictionary of current features
        history_df: DataFrame of historical data
    
    Returns:
        Dictionary with added time-series features
    """
    # Create DataFrame from current row
    current_df = pd.DataFrame([current_row])
    current_df['timestamp'] = pd.to_datetime(current_df['timestamp'])
    
    # Combine with history
    if not history_df.empty:
        df = pd.concat([history_df, current_df], ignore_index=True)
    else:
        df = current_df
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Compute time-series features
    row = df.iloc[-1].copy()
    
    # Wave height features
    if 'marine_wave_height' in df.columns:
        wave_6h_avg = df['marine_wave_height'].tail(6).mean()
        row['wave_6h_avg'] = wave_6h_avg
        row['wave_delta'] = row['marine_wave_height'] - wave_6h_avg
    else:
        row['wave_6h_avg'] = 0
        row['wave_delta'] = 0
    
    # Pressure difference
    if 'weather_airPressure' in df.columns and len(df) > 1:
        row['pressure_diff'] = row['weather_airPressure'] - df['weather_airPressure'].iloc[-2]
    else:
        row['pressure_diff'] = 0
    
    # News spike
    if 'news_news_total_articles' in df.columns:
        news_rolling = df['news_news_total_articles'].tail(24).mean()
        row['news_rolling'] = news_rolling
        row['news_spike'] = row['news_news_total_articles'] - news_rolling
    else:
        row['news_rolling'] = 0
        row['news_spike'] = 0
    
    # Ensure all features exist
    required_features = ['wave_delta', 'pressure_diff', 'news_spike']
    for feat in required_features:
        if feat not in row or pd.isna(row[feat]):
            row[feat] = 0
    
    return row.to_dict()

def compute_batch_features(df):
    """
    Compute time-series features for a batch of data.
    Used for training data preprocessing.
    
    Args:
        df: DataFrame with timestamp column
    
    Returns:
        DataFrame with added time-series features
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=["city", "timestamp"])
    
    # Group by city for time-series calculations
    df['wave_6h_avg'] = df.groupby('city')['marine_wave_height'].transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )
    df['wave_delta'] = df['marine_wave_height'] - df['wave_6h_avg']
    
    df['pressure_diff'] = df.groupby('city')['weather_airPressure'].diff().fillna(0)
    
    df['news_rolling'] = df.groupby('city')['news_news_total_articles'].transform(
        lambda x: x.rolling(24, min_periods=1).mean()
    )
    df['news_spike'] = df['news_news_total_articles'] - df['news_rolling']
    
    # Fill NaN values
    df['wave_delta'] = df['wave_delta'].fillna(0)
    df['pressure_diff'] = df['pressure_diff'].fillna(0)
    df['news_spike'] = df['news_spike'].fillna(0)
    
    return df

if __name__ == "__main__":
    # Test the module
    print("Testing time_features module...")
    
    # Load history
    history = load_history()
    print(f"Loaded {len(history)} historical rows")
    
    # Test with a sample row
    sample_row = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'city': 'Tripoli',
        'marine_wave_height': 2.5,
        'weather_airPressure': 1013,
        'news_news_total_articles': 10
    }
    
    result = compute_realtime_features(sample_row, history)
    print(f"\nSample row with time features:")
    for key in ['wave_delta', 'pressure_diff', 'news_spike']:
        print(f"  {key}: {result.get(key, 0):.4f}")
