#!/usr/bin/env python3
"""
V2 Risk Engine with Hybrid ML + Anomaly + Trend Scoring.
Upgrades from rule-based to learned AI with ensemble fusion.
"""

import joblib
import numpy as np
import pandas as pd
import os

MODEL_DIR = "/var/www/html/ml/models_v2"
OLD_MODEL_DIR = "/var/www/html/ml/models"

class RiskEngineV2:
    """Next-gen hybrid risk engine with ML, anomaly, and trend fusion."""
    
    def __init__(self):
        """Initialize V2 risk engine with models."""
        try:
            self.xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
            self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
            self.features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
            print("✅ V2 models loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load V2 models: {e}")
            print("Falling back to V1 models...")
            self.xgb_model = None
            self.scaler = None
            self.features = None
        
        # Load old Isolation Forest for anomaly detection
        try:
            self.iso_forest = joblib.load(os.path.join(OLD_MODEL_DIR, "isolation_forest.pkl"))
            print("✅ Isolation Forest loaded")
        except Exception as e:
            print(f"Warning: Could not load Isolation Forest: {e}")
            self.iso_forest = None
        
        # Risk history for stability tracking
        self.risk_history = {}  # city -> list of recent risk scores
    
    def predict_ml_risk(self, row):
        """
        Predict disaster probability using XGBoost model.
        
        Args:
            row: Dictionary of features
        
        Returns:
            Probability of disaster (0-1)
        """
        if self.xgb_model is None or self.scaler is None or self.features is None:
            return 0.0
        
        try:
            # Extract features in correct order using DataFrame to avoid scaler warning
            X = pd.DataFrame([[row.get(f, 0) for f in self.features]], columns=self.features)
            X_scaled = self.scaler.transform(X)
            
            # Get probability of class 1 (disaster)
            prob = self.xgb_model.predict_proba(X_scaled)[0][1]
            return float(prob)
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return 0.0
    
    def predict_anomaly(self, row):
        """
        Predict anomaly using Isolation Forest.
        
        Args:
            row: Dictionary of features
        
        Returns:
            Anomaly score (0 or 1)
        """
        if self.iso_forest is None:
            return 0.0
        
        try:
            # Use old features for anomaly detection
            old_features = [
                'weather_airPressure', 'weather_airQuality', 'weather_weather_risk_score', 'weather_weather_risk_level',
                'marine_sea_level_anomaly', 'marine_swell_wave_height', 'marine_wave_height', 'marine_wave_period',
                'marine_wind_speed', 'marine_wind_wave_height',
                'news_news_critical', 'news_news_high', 'news_news_medium', 'news_news_low',
                'news_news_total_articles', 'news_news_risk_score',
                'geo_lat', 'geo_lon', 'geo_elevation', 'geo_distance_to_coast'
            ]
            
            # Use DataFrame to avoid scaler warning
            X = pd.DataFrame([[row.get(f, 0) for f in old_features]], columns=old_features)
            
            # Load old scaler
            old_scaler = joblib.load(os.path.join(OLD_MODEL_DIR, "scaler.pkl"))
            X_scaled = old_scaler.transform(X)
            
            anomaly_pred = self.iso_forest.predict(X_scaled)[0]
            return 1.0 if anomaly_pred == -1 else 0.0
        except Exception as e:
            print(f"Error in anomaly prediction: {e}")
            return 0.0
    
    def compute_trend_score(self, row):
        """
        Compute trend score based on time-series features.
        
        Args:
            row: Dictionary of features with time-series data
        
        Returns:
            Trend score (0-1)
        """
        score = 0.0
        
        # Wave height increase (sudden storm surge)
        wave_delta = row.get('wave_delta', 0)
        if wave_delta > 0.5:
            score += 0.4
        elif wave_delta > 0.2:
            score += 0.2
        
        # Pressure drop (storm approaching)
        pressure_diff = row.get('pressure_diff', 0)
        if pressure_diff < -5:  # Significant pressure drop
            score += 0.4
        elif pressure_diff < -2:
            score += 0.2
        
        # News spike (early warning)
        news_spike = row.get('news_spike', 0)
        if news_spike > 10:
            score += 0.2
        elif news_spike > 5:
            score += 0.1
        
        return min(score, 1.0)
    
    def compute_geo_risk(self, row):
        """
        Compute geographic risk based on location.
        
        Args:
            row: Dictionary with geolocation features
        
        Returns:
            Geo risk score (0-1)
        """
        distance_to_coast = row.get('geo_distance_to_coast', 100)
        elevation = row.get('geo_elevation', 100)
        
        # Coastal areas have higher flood risk
        coastal_risk = 0.5 if distance_to_coast < 5 else 0.2
        
        # Lowland areas have higher flood risk
        lowland_risk = 0.5 if elevation < 50 else 0.2
        
        geo_risk = (coastal_risk + lowland_risk) / 2
        return min(geo_risk, 1.0)
    
    def compute_final_risk(self, ml_score, anomaly_score, trend_score, geo_risk=0.0):
        """
        Compute final hybrid risk score using ensemble fusion.
        
        Args:
            ml_score: XGBoost probability (0-1)
            anomaly_score: Isolation Forest anomaly (0 or 1)
            trend_score: Trend analysis score (0-1)
            geo_risk: Geographic risk score (0-1)
        
        Returns:
            Final risk score (0-1)
        """
        # Weighted ensemble
        # ML gets highest weight (learned from real data)
        # Anomaly provides anomaly detection
        # Trend provides temporal context
        # Geo provides spatial context
        final_score = (
            0.5 * ml_score +
            0.2 * anomaly_score +
            0.2 * trend_score +
            0.1 * geo_risk
        )
        
        return min(final_score, 1.0)
    
    def classify(self, score, confidence):
        """
        Classify risk score into risk level with confidence-based alert logic.
        
        Args:
            score: Risk score (0-1)
            confidence: Confidence score (0-1)
        
        Returns:
            Risk level string with confidence suffix
        """
        if score >= 0.7:
            if confidence > 0.8:
                return "HIGH_CONFIRMED"
            else:
                return "HIGH_UNCERTAIN"
        elif score >= 0.4:
            if confidence > 0.8:
                return "MEDIUM_CONFIRMED"
            else:
                return "MEDIUM_UNCERTAIN"
        return "LOW"
    
    def compute_confidence(self, ml_score):
        """
        Compute confidence and uncertainty from ML probability.
        
        Args:
            ml_score: ML probability (0-1)
        
        Returns:
            Dictionary with confidence and uncertainty
        """
        # Confidence: how far from 0.5 (decision boundary)
        confidence = abs(ml_score - 0.5) * 2
        uncertainty = 1 - confidence
        
        return {
            'confidence': float(confidence),
            'uncertainty': float(uncertainty)
        }
    
    def update_risk_history(self, city, risk_score):
        """
        Update risk history for a city and compute trend direction.
        
        Args:
            city: City name
            risk_score: Current risk score
        
        Returns:
            Trend direction string
        """
        if city not in self.risk_history:
            self.risk_history[city] = []
        
        self.risk_history[city].append(risk_score)
        
        # Keep only last 3 predictions
        if len(self.risk_history[city]) > 3:
            self.risk_history[city].pop(0)
        
        # Compute trend direction
        if len(self.risk_history[city]) >= 2:
            recent_avg = sum(self.risk_history[city][-2:]) / 2
            if risk_score > recent_avg + 0.1:
                return "rising"
            elif risk_score < recent_avg - 0.1:
                return "falling"
        
        return "stable"
    
    def should_trigger_alert(self, risk_result, threshold=0.7):
        """
        Determine if an alert should be triggered.
        
        Args:
            risk_result: Dictionary with risk scores
            threshold: Alert threshold (default 0.7)
        
        Returns:
            Boolean indicating if alert should trigger
        """
        return risk_result['total_risk_score'] >= threshold
    
    def predict(self, row, city='unknown'):
        """
        Full prediction pipeline with all components.
        
        Args:
            row: Dictionary of features (should include time-series features)
            city: City name for risk history tracking
        
        Returns:
            Dictionary with risk scores and classification
        """
        # ML prediction
        ml_score = self.predict_ml_risk(row)
        
        # Anomaly prediction
        anomaly_score = self.predict_anomaly(row)
        
        # Trend analysis
        trend_score = self.compute_trend_score(row)
        
        # Geo risk
        geo_risk = self.compute_geo_risk(row)
        
        # Compute confidence and uncertainty
        confidence_metrics = self.compute_confidence(ml_score)
        
        # Final hybrid score
        final_score = self.compute_final_risk(ml_score, anomaly_score, trend_score, geo_risk)
        
        # Classification with confidence-based alert logic
        risk_level = self.classify(final_score, confidence_metrics['confidence'])
        
        # Improved early warning detection with anomaly
        early_warning = (trend_score > 0.6 and anomaly_score == 1)
        
        # Update risk history and get trend direction
        trend_direction = self.update_risk_history(city, final_score)
        
        return {
            'total_risk_score': final_score,
            'risk_level': risk_level,
            'component_scores': {
                'ml': ml_score,
                'anomaly': anomaly_score,
                'trend': trend_score
            },
            'weights': {
                'ml': 0.5,
                'anomaly': 0.2,
                'trend': 0.2,
                'geo': 0.1
            },
            'confidence': confidence_metrics['confidence'],
            'uncertainty': confidence_metrics['uncertainty'],
            'early_warning': early_warning,
            'trend_direction': trend_direction
        }

if __name__ == "__main__":
    # Test the risk engine
    print("Testing RiskEngineV2...")
    
    engine = RiskEngineV2()
    
    # Test with sample data
    sample_row = {
        'weather_airPressure': 1013,
        'weather_airQuality': 50,
        'weather_weather_risk_score': 0.3,
        'weather_weather_risk_level': 1,
        'marine_wave_height': 2.5,
        'marine_wind_speed': 10,
        'marine_sea_level_anomaly': 0.1,
        'news_news_risk_score': 0.2,
        'news_alert_intensity': 2,
        'geo_lat': 32.8872,
        'geo_lon': 13.1913,
        'geo_elevation': 25,
        'geo_distance_to_coast': 0.0,
        'wave_delta': 0.3,
        'pressure_diff': -1,
        'news_spike': 3
    }
    
    result = engine.predict(sample_row, city='Tripoli')
    print(f"\nPrediction result:")
    print(f"  Total Risk Score: {result['total_risk_score']:.4f}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Components:")
    print(f"    ML: {result['component_scores']['ml']:.4f}")
    print(f"    Anomaly: {result['component_scores']['anomaly']:.4f}")
    print(f"    Trend: {result['component_scores']['trend']:.4f}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Uncertainty: {result['uncertainty']:.4f}")
    print(f"  Early Warning: {result['early_warning']}")
    print(f"  Trend Direction: {result['trend_direction']}")
