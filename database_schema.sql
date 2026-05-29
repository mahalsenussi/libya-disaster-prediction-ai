-- Database schema for disaster prediction system logging and monitoring
-- This schema supports SQLite, PostgreSQL, or MySQL

-- Cities table
CREATE TABLE IF NOT EXISTS cities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL UNIQUE,
    lat DECIMAL(10, 6) NOT NULL,
    lon DECIMAL(10, 6) NOT NULL,
    elevation DECIMAL(10, 2),
    distance_to_coast DECIMAL(10, 2),
    region VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Risk history table - stores all risk predictions
CREATE TABLE IF NOT EXISTS risk_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    risk_score DECIMAL(5, 4) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    weather_score DECIMAL(5, 4),
    sea_score DECIMAL(5, 4),
    news_score DECIMAL(5, 4),
    anomaly_score DECIMAL(5, 4),
    alert_triggered BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (city_id) REFERENCES cities(id)
);

-- Alerts table - stores triggered alerts
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    risk_score DECIMAL(5, 4) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    alert_status VARCHAR(20) DEFAULT 'ACTIVE', -- ACTIVE, ACKNOWLEDGED, RESOLVED
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP,
    notes TEXT,
    FOREIGN KEY (city_id) REFERENCES cities(id)
);

-- Data collection log table
CREATE TABLE IF NOT EXISTS data_collection_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL,
    cities_collected INTEGER NOT NULL,
    weather_api_success BOOLEAN,
    marine_api_success BOOLEAN,
    news_api_success BOOLEAN,
    collection_duration_seconds DECIMAL(10, 2),
    error_message TEXT
);

-- Model performance metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    timestamp TIMESTAMP NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10, 4),
    notes TEXT
);

-- Disaster events table - for ground truth labels
CREATE TABLE IF NOT EXISTS disaster_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id INTEGER NOT NULL,
    event_type VARCHAR(50) NOT NULL, -- flood, storm, earthquake, etc.
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    severity VARCHAR(20), -- LOW, MEDIUM, HIGH, CRITICAL
    affected_population INTEGER,
    confirmed BOOLEAN DEFAULT FALSE,
    source VARCHAR(100), -- manual, news, official_report
    notes TEXT,
    FOREIGN KEY (city_id) REFERENCES cities(id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_risk_history_city_timestamp ON risk_history(city_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_risk_history_timestamp ON risk_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_city_timestamp ON alerts(city_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(alert_status);
CREATE INDEX IF NOT EXISTS idx_disaster_events_city_time ON disaster_events(city_id, start_time);

-- Sample data for cities
INSERT OR IGNORE INTO cities (name, lat, lon, elevation, distance_to_coast, region) VALUES
('Tripoli', 32.8872, 13.1913, 25, 0.0, 'northwest'),
('Benghazi', 32.1165, 20.0667, 30, 0.0, 'northeast'),
('Misrata', 32.3753, 15.0927, 10, 0.0, 'northwest'),
('Al Khums', 32.6519, 14.2566, 5, 0.0, 'northwest'),
('Zuwara', 32.9319, 12.0456, 3, 0.0, 'northwest'),
('Derna', 32.7556, 22.6333, 15, 0.0, 'northeast'),
('Tobruk', 32.0833, 23.9667, 20, 0.0, 'northeast'),
('Al Bayda', 32.7625, 21.7667, 600, 15.0, 'northeast'),
('Sirte', 31.2083, 16.5833, 10, 0.0, 'north'),
('Zawiya', 32.7636, 12.7275, 8, 0.0, 'northwest'),
('Ajdabiya', 30.7536, 20.9333, 50, 20.0, 'northeast'),
('Sabha', 27.0167, 14.4333, 250, 500.0, 'south'),
('Ghat', 24.9667, 10.1667, 500, 800.0, 'southwest'),
('Ubari', 26.5833, 12.8167, 350, 600.0, 'south'),
('Murzuq', 25.9167, 13.9167, 300, 700.0, 'south');
