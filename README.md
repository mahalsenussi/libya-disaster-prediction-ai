# Libya Disaster Prediction AI System

A production-grade AI-powered disaster prediction system for Libya, featuring hybrid ensemble learning, time-series intelligence, and real-time risk assessment.

## 🚀 Features

- **Hybrid Risk Engine**: Combines XGBoost ML, Isolation Forest anomaly detection, trend analysis, and geographic risk
- **Time-Series Intelligence**: Rolling averages, deltas, and spike detection for early warning
- **Probability Calibration**: Isotonic calibration for realistic confidence estimates
- **Confidence-Based Alerts**: HIGH_CONFIRMED vs HIGH_UNCERTAIN classification
- **Risk Stability Tracking**: Monitors risk trends (rising/falling/stable) across predictions
- **Early Warning Detection**: Pre-disaster pattern recognition
- **Real-Time Feedback Loop**: API for collecting real-world event verification
- **Geographic Awareness**: Coastal proximity and elevation-based risk factors

## 📊 Architecture

```
Data Layer
├── Weather API (air pressure, quality, risk scores)
├── Marine API (wave height, wind speed, sea level)
├── News API (article counts, keyword analysis)
└── Geolocation (lat, lon, elevation, distance to coast)

Processing Layer
├── Time-series features (wave_delta, pressure_diff, news_spike)
└── Feature scaling and normalization

AI Layer
├── XGBoost Classifier (isotonic calibrated)
├── Isolation Forest (anomaly detection)
└── Trend Analysis (temporal patterns)

Fusion Layer
└── Hybrid Ensemble: 50% ML + 20% Anomaly + 20% Trend + 10% Geo

Output Layer
├── Risk Score (0-1)
├── Risk Level (HIGH_CONFIRMED, HIGH_UNCERTAIN, MEDIUM_CONFIRMED, MEDIUM_UNCERTAIN, LOW)
├── Confidence & Uncertainty
├── Trend Direction (rising/falling/stable)
└── Early Warning Flag
```

## 📁 Project Structure

```
/var/www/html/ml
├── Core System
│   ├── alert_api.py              # Flask API with V2 risk engine
│   ├── collect_data.py           # Data collection from APIs
│   ├── test_prediction.py        # Standalone prediction testing
│   └── pipeline.py              # Real-time data collection loop
│
├── V2 AI Components
│   ├── train_model_v2.py         # XGBoost training with calibration
│   ├── risk_engine_v2.py         # Hybrid risk engine (ML + anomaly + trend + geo)
│   ├── time_features.py          # Time-series feature computation
│   └── city_geolocation.py       # Libyan city geolocation data
│
├── V1 Legacy Components
│   ├── train_model.py            # Original Random Forest training
│   └── risk_engine.py            # Original risk engine
│
├── Data Processing
│   ├── emdat_processor.py        # EM-DAT disaster data processing
│   ├── event_matcher.py          # Event matching for labeling
│   └── synthetic_trainer.py      # Synthetic data generation with realism
│
├── Database
│   ├── db_logger.py              # Database logging utilities
│   └── database_schema.sql      # Database schema definition
│
├── Data/
│   ├── libya_disasters.csv      # EM-DAT processed disaster data
│   └── disaster_data_labeled.csv # Training dataset (synthetic + real)
│
├── Models/
│   ├── isolation_forest.pkl      # V1 anomaly detection
│   ├── random_forest.pkl         # V1 classifier
│   └── scaler.pkl                # V1 feature scaler
│
├── Models_v2/
│   ├── xgb_model.pkl             # V2 calibrated XGBoost model
│   ├── scaler.pkl                # V2 feature scaler
│   └── features.pkl              # V2 feature list
│
└── Documentation
    ├── README.md                 # This file
    ├── UPGRADE_SUMMARY.md        # V1 upgrade summary
    ├── EMDAT_INTEGRATION_SUMMARY.md # EM-DAT integration summary
    └── V2_UPGRADE_SUMMARY.md    # V2 upgrade summary
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd ml

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost flask requests joblib
```

## 🚀 Quick Start

### 1. Generate Training Data

```bash
# Process EM-DAT disaster data
python emdat_processor.py

# Generate synthetic training dataset with realistic noise
python synthetic_trainer.py
```

### 2. Train V2 Model

```bash
# Train calibrated XGBoost model
python train_model_v2.py
```

### 3. Start API

```bash
# Start the Flask API
python alert_api.py
```

The API runs on `http://localhost:5002` by default.

## 📡 API Endpoints

### Predict Risk

```bash
GET /api/predict_risk/<city>
```

**Response:**
```json
{
  "city": "Tripoli",
  "timestamp": "2026-05-29T10:00:00Z",
  "risk_score": 0.7553,
  "risk_level": "HIGH_CONFIRMED",
  "component_scores": {
    "ml": 0.9306,
    "anomaly": 1.0000,
    "trend": 0.2000
  },
  "weights": {
    "ml": 0.5,
    "anomaly": 0.2,
    "trend": 0.2,
    "geo": 0.1
  },
  "confidence": 0.8611,
  "uncertainty": 0.1389,
  "early_warning": false,
  "trend_direction": "stable"
}
```

### Submit Feedback

```bash
POST /api/feedback
Content-Type: application/json

{
  "city": "Derna",
  "event_happened": true,
  "severity": 3,
  "false_alarm": false,
  "timestamp": "2026-05-29T10:00:00Z",
  "notes": "Flood after heavy rainfall"
}
```

### Get Alerts

```bash
GET /api/alerts
```

### Clear Alerts

```bash
POST /api/alerts/clear
```

## 🧪 Testing

```bash
# Test prediction standalone
python test_prediction.py

# Test risk engine directly
python risk_engine_v2.py
```

## 📈 Model Performance

**Current V2 Model (55,023 training samples):**
- Accuracy: 0.99
- Disaster Precision: 0.94
- Disaster Recall: 1.00
- F1-Score: 0.97

**Feature Importance:**
- weather_risk_level: 26.97%
- pressure_diff: 15.67%
- news_alert_intensity: 10.15%
- news_risk_score: 10.04%
- geo_lat: 7.27%
- geo_elevation: 7.13%
- geo_distance_to_coast: 7.10%
- news_spike: 6.88%

## 🔬 Advanced Features

### Probability Calibration

Uses isotonic regression to calibrate XGBoost probabilities, reducing overconfidence and providing more realistic risk estimates.

### Confidence-Based Alert Logic

- **HIGH_CONFIRMED**: Risk ≥ 0.7 AND confidence > 0.8 → Send response team
- **HIGH_UNCERTAIN**: Risk ≥ 0.7 AND confidence ≤ 0.8 → Monitor closely
- **MEDIUM_CONFIRMED**: Risk ≥ 0.4 AND confidence > 0.8 → Prepare
- **MEDIUM_UNCERTAIN**: Risk ≥ 0.4 AND confidence ≤ 0.8 → Monitor
- **LOW**: Risk < 0.4 → Normal operations

### Early Warning Detection

Triggers when:
- Trend score > 0.6 (rising pattern)
- Anomaly score == 1 (unusual conditions detected)

This enables pre-disaster detection rather than just disaster detection.

### Risk Stability Tracking

Tracks last 3 predictions per city to compute:
- **Rising**: Risk increasing by > 0.1
- **Falling**: Risk decreasing by > 0.1
- **Stable**: Risk within ± 0.1

## 🔄 Continuous Learning

The system supports real-world feedback collection for continuous model improvement:

1. Deploy API and collect predictions
2. Submit feedback when events occur
3. Retrain model weekly with hybrid dataset (synthetic + real)
4. Monitor model drift and recalibrate as needed

## 📝 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
flask>=2.0.0
requests>=2.26.0
joblib>=1.0.0
```

## 🚧 Production Deployment

For production deployment:

1. Use Gunicorn or uWSGI instead of Flask dev server
2. Implement proper authentication
3. Add rate limiting
4. Set up database for persistent logging
5. Configure monitoring and alerting
6. Implement model versioning
7. Add automated retraining pipeline

## 📄 License

This project is developed for disaster prediction and humanitarian assistance purposes.

## 🤝 Contributing

Contributions are welcome, especially:
- Additional data sources
- Model improvements
- Feature engineering
- Documentation enhancements

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Note**: This system is designed for disaster prediction and should be used as a decision support tool. Always verify predictions with local authorities and expert judgment.
