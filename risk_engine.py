#!/usr/bin/env python3
"""
Hybrid Risk Scoring Engine for disaster prediction.
Combines multiple risk factors into a unified risk score.
"""

import numpy as np

class RiskEngine:
    """Hybrid risk scoring engine combining weather, sea, news, and anomaly scores."""
    
    def __init__(self, weights=None):
        """
        Initialize risk engine with custom weights or defaults.
        
        Default weights:
        - weather_risk: 0.5
        - sea_risk: 0.2
        - news_risk: 0.2
        - anomaly_risk: 0.1
        """
        if weights is None:
            self.weights = {
                'weather_risk': 0.5,
                'sea_risk': 0.2,
                'news_risk': 0.2,
                'anomaly_risk': 0.1
            }
        else:
            self.weights = weights
    
    def normalize_score(self, score, min_val, max_val):
        """Normalize a score to 0-1 range."""
        if max_val == min_val:
            return 0.0
        return (score - min_val) / (max_val - min_val)
    
    def calculate_weather_risk(self, features):
        """Calculate weather risk score from features."""
        # Weather risk level (0-3)
        weather_risk_level = features.get('weather_weather_risk_level', 0)
        weather_risk_score = features.get('weather_weather_risk_score', 0.0)
        
        # Normalize risk level to 0-1
        normalized_level = self.normalize_score(weather_risk_level, 0, 3)
        
        # Normalize risk score (assuming max 1.0)
        normalized_score = min(weather_risk_score / 1.0, 1.0)
        
        # Combine: 70% weight to level, 30% to score
        weather_risk = 0.7 * normalized_level + 0.3 * normalized_score
        return weather_risk
    
    def calculate_sea_risk(self, features):
        """Calculate sea/marine risk score from features."""
        wave_height = features.get('marine_wave_height', 0.0)
        wind_speed = features.get('marine_wind_speed', 0.0)
        sea_level_anomaly = abs(features.get('marine_sea_level_anomaly', 0.0))
        
        # Normalize wave height (dangerous above 2m)
        wave_risk = min(wave_height / 2.0, 1.0)
        
        # Normalize wind speed (dangerous above 15 m/s)
        wind_risk = min(wind_speed / 15.0, 1.0)
        
        # Normalize sea level anomaly (dangerous above 0.5m)
        sea_risk = min(sea_level_anomaly / 0.5, 1.0)
        
        # Combine: 40% wave, 40% wind, 20% sea level
        sea_risk_score = 0.4 * wave_risk + 0.4 * wind_risk + 0.2 * sea_risk
        return sea_risk_score
    
    def calculate_news_risk(self, features):
        """Calculate news risk score from features."""
        news_risk_score = features.get('news_news_risk_score', 0.0)
        alert_intensity = features.get('news_alert_intensity', 0.0)
        
        # Normalize news risk score (max 3.0)
        normalized_news_score = min(news_risk_score / 3.0, 1.0)
        
        # Normalize alert intensity (dangerous above 10)
        normalized_alert = min(alert_intensity / 10.0, 1.0)
        
        # Combine: 60% news score, 40% alert intensity
        news_risk = 0.6 * normalized_news_score + 0.4 * normalized_alert
        return news_risk
    
    def calculate_anomaly_risk(self, anomaly_score):
        """Calculate anomaly risk score (0 or 1)."""
        return float(anomaly_score)
    
    def calculate_total_risk(self, features, anomaly_score=0):
        """
        Calculate total hybrid risk score.
        
        Args:
            features: Dictionary of feature values
            anomaly_score: Anomaly score from Isolation Forest (0 or 1)
        
        Returns:
            Dictionary with total risk score and component scores
        """
        weather_risk = self.calculate_weather_risk(features)
        sea_risk = self.calculate_sea_risk(features)
        news_risk = self.calculate_news_risk(features)
        anomaly_risk = self.calculate_anomaly_risk(anomaly_score)
        
        # Calculate weighted total
        total_risk = (
            self.weights['weather_risk'] * weather_risk +
            self.weights['sea_risk'] * sea_risk +
            self.weights['news_risk'] * news_risk +
            self.weights['anomaly_risk'] * anomaly_risk
        )
        
        # Determine risk level
        if total_risk >= 0.7:
            risk_level = 'HIGH'
        elif total_risk >= 0.4:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'total_risk_score': total_risk,
            'risk_level': risk_level,
            'component_scores': {
                'weather_risk': weather_risk,
                'sea_risk': sea_risk,
                'news_risk': news_risk,
                'anomaly_risk': anomaly_risk
            },
            'weights': self.weights
        }
    
    def should_trigger_alert(self, risk_result, threshold=0.7):
        """Determine if an alert should be triggered based on risk score."""
        return risk_result['total_risk_score'] >= threshold
