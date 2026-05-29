#!/usr/bin/env python3
"""
Database logger for disaster prediction system.
Handles logging of risk predictions, alerts, and system metrics.
"""

import sqlite3
import os
from datetime import datetime, timezone
from contextlib import contextmanager

DB_PATH = "/var/www/html/ml/disaster_prediction.db"

@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def init_database():
    """Initialize database with schema."""
    if not os.path.exists(DB_PATH):
        # Create database file
        open(DB_PATH, 'a').close()
    
    with get_db_connection() as conn:
        with open('/var/www/html/ml/database_schema.sql', 'r') as f:
            schema = f.read()
            conn.executescript(schema)
    print("Database initialized successfully.")

def log_risk_prediction(city_id, risk_result, features):
    """Log a risk prediction to the database."""
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO risk_history 
            (city_id, timestamp, risk_score, risk_level, weather_score, sea_score, news_score, anomaly_score, alert_triggered)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            city_id,
            datetime.now(timezone.utc).isoformat(),
            risk_result['total_risk_score'],
            risk_result['risk_level'],
            risk_result['component_scores']['weather_risk'],
            risk_result['component_scores']['sea_risk'],
            risk_result['component_scores']['news_risk'],
            risk_result['component_scores']['anomaly_risk'],
            risk_result['total_risk_score'] >= 0.7
        ))

def log_alert(city_id, risk_result):
    """Log an alert to the database."""
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO alerts 
            (city_id, timestamp, risk_score, risk_level, alert_status)
            VALUES (?, ?, ?, ?, ?)
        """, (
            city_id,
            datetime.now(timezone.utc).isoformat(),
            risk_result['total_risk_score'],
            risk_result['risk_level'],
            'ACTIVE'
        ))

def log_data_collection(cities_count, weather_success, marine_success, news_success, duration, error=None):
    """Log data collection attempt."""
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO data_collection_log 
            (timestamp, cities_collected, weather_api_success, marine_api_success, news_api_success, collection_duration_seconds, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            cities_count,
            weather_success,
            marine_success,
            news_success,
            duration,
            error
        ))

def get_city_id(city_name):
    """Get city ID from database."""
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT id FROM cities WHERE name = ?", (city_name,))
        row = cursor.fetchone()
        return row['id'] if row else None

def get_recent_alerts(limit=100):
    """Get recent alerts from database."""
    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT a.*, c.name as city_name 
            FROM alerts a
            JOIN cities c ON a.city_id = c.id
            ORDER BY a.timestamp DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

def get_risk_history(city_name, limit=100):
    """Get risk history for a city."""
    city_id = get_city_id(city_name)
    if not city_id:
        return []
    
    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT * FROM risk_history
            WHERE city_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (city_id, limit))
        return [dict(row) for row in cursor.fetchall()]

def log_disaster_event(city_name, event_type, start_time, severity, source, notes=None):
    """Log a disaster event for ground truth labeling."""
    city_id = get_city_id(city_name)
    if not city_id:
        print(f"City {city_name} not found in database.")
        return
    
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO disaster_events 
            (city_id, event_type, start_time, severity, source, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            city_id,
            event_type,
            start_time,
            severity,
            source,
            notes
        ))

if __name__ == "__main__":
    # Initialize database
    init_database()
    print("Database setup complete.")
