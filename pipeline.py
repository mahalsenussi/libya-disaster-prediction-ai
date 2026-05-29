#!/usr/bin/env python3
"""
Real-time data collection and prediction pipeline.
Runs continuous loop to collect data, predict risks, and trigger alerts.
"""

import time
import signal
import sys
from datetime import datetime, timezone
from collect_data import collect_data, save_to_csv, CITIES
from db_logger import log_data_collection, get_city_id, log_risk_prediction, log_alert
from risk_engine import RiskEngine
import joblib
import pandas as pd
import numpy as np
import os

# Configuration
COLLECTION_INTERVAL_MINUTES = 15  # Collect data every 15 minutes
CSV_PATH = "/var/www/html/ml/data/disaster_data.csv"
MODEL_DIR = "/var/www/html/ml/models"

# Global flag for graceful shutdown
running = True

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running
    print("\nShutdown signal received. Stopping pipeline...")
    running = False

def load_models():
    """Load trained models."""
    try:
        iso_forest = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))
        return iso_forest, scaler, feature_cols
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

def predict_risk_for_city(city, row, iso_forest, scaler, feature_cols, risk_engine):
    """Predict risk for a single city."""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([row])
        
        # Extract features
        X = df[feature_cols].copy()
        X = X.fillna(X.mean())
        X_scaled = scaler.transform(X)
        
        # Get anomaly score
        anomaly_pred = iso_forest.predict(X_scaled)[0]
        anomaly_score = 1 if anomaly_pred == -1 else 0
        
        # Calculate hybrid risk score
        risk_result = risk_engine.calculate_total_risk(row, anomaly_score)
        
        return risk_result
    except Exception as e:
        print(f"Error predicting risk for {city}: {e}")
        return None

def run_pipeline_cycle(iso_forest, scaler, feature_cols, risk_engine):
    """Run one complete pipeline cycle."""
    print(f"\n=== Pipeline Cycle Started at {datetime.now(timezone.utc).isoformat()} ===")
    
    start_time = time.time()
    
    # Collect data
    print("Collecting data...")
    rows = collect_data()
    
    if not rows:
        print("No data collected. Skipping this cycle.")
        return
    
    # Save to CSV
    save_to_csv(rows, CSV_PATH)
    
    # Log data collection
    collection_duration = time.time() - start_time
    log_data_collection(
        cities_count=len(rows),
        weather_success=True,  # Simplified - could track per API
        marine_success=True,
        news_success=True,
        duration=collection_duration
    )
    
    # Predict risks for each city
    print("Predicting risks...")
    risk_engine = RiskEngine()
    
    for row in rows:
        city = row['city']
        city_id = get_city_id(city)
        
        if not city_id:
            print(f"City {city} not found in database. Skipping risk prediction.")
            continue
        
        risk_result = predict_risk_for_city(city, row, iso_forest, scaler, feature_cols, risk_engine)
        
        if risk_result:
            # Log risk prediction
            log_risk_prediction(city_id, risk_result, row)
            
            # Log alert if triggered
            if risk_result['total_risk_score'] >= 0.7:
                log_alert(city_id, risk_result)
                print(f"ALERT: {city} - {risk_result['risk_level']} risk (score: {risk_result['total_risk_score']:.2f})")
            else:
                print(f"{city} - {risk_result['risk_level']} risk (score: {risk_result['total_risk_score']:.2f})")
    
    print(f"=== Pipeline Cycle Completed in {time.time() - start_time:.2f}s ===")

def main():
    """Main pipeline loop."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize database
    from db_logger import init_database
    init_database()
    
    # Load models
    print("Loading models...")
    iso_forest, scaler, feature_cols = load_models()
    
    if iso_forest is None:
        print("Failed to load models. Exiting.")
        sys.exit(1)
    
    # Initialize risk engine
    risk_engine = RiskEngine()
    
    print(f"Pipeline started. Collecting data every {COLLECTION_INTERVAL_MINUTES} minutes.")
    print("Press Ctrl+C to stop.")
    
    # Run initial cycle
    run_pipeline_cycle(iso_forest, scaler, feature_cols, risk_engine)
    
    # Main loop
    while running:
        try:
            # Sleep until next cycle
            time.sleep(COLLECTION_INTERVAL_MINUTES * 60)
            
            if not running:
                break
            
            # Run pipeline cycle
            run_pipeline_cycle(iso_forest, scaler, feature_cols, risk_engine)
            
        except Exception as e:
            print(f"Error in pipeline cycle: {e}")
            # Continue running despite errors
    
    print("Pipeline stopped.")

if __name__ == "__main__":
    main()
