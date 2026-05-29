#!/usr/bin/env python3
"""
Geolocation data for Libyan cities.
Includes latitude, longitude, elevation, and distance to coast.
"""

CITY_GEOLOCATION = {
    'Tripoli': {
        'lat': 32.8872,
        'lon': 13.1913,
        'elevation': 25,  # meters
        'distance_to_coast': 0.0,  # coastal city
        'region': 'northwest'
    },
    'Benghazi': {
        'lat': 32.1165,
        'lon': 20.0667,
        'elevation': 30,
        'distance_to_coast': 0.0,
        'region': 'northeast'
    },
    'Misrata': {
        'lat': 32.3753,
        'lon': 15.0927,
        'elevation': 10,
        'distance_to_coast': 0.0,
        'region': 'northwest'
    },
    'Al Khums': {
        'lat': 32.6519,
        'lon': 14.2566,
        'elevation': 5,
        'distance_to_coast': 0.0,
        'region': 'northwest'
    },
    'Zuwara': {
        'lat': 32.9319,
        'lon': 12.0456,
        'elevation': 3,
        'distance_to_coast': 0.0,
        'region': 'northwest'
    },
    'Derna': {
        'lat': 32.7556,
        'lon': 22.6333,
        'elevation': 15,
        'distance_to_coast': 0.0,
        'region': 'northeast'
    },
    'Tobruk': {
        'lat': 32.0833,
        'lon': 23.9667,
        'elevation': 20,
        'distance_to_coast': 0.0,
        'region': 'northeast'
    },
    'Al Bayda': {
        'lat': 32.7625,
        'lon': 21.7667,
        'elevation': 600,
        'distance_to_coast': 15.0,  # km
        'region': 'northeast'
    },
    'Sirte': {
        'lat': 31.2083,
        'lon': 16.5833,
        'elevation': 10,
        'distance_to_coast': 0.0,
        'region': 'north'
    },
    'Zawiya': {
        'lat': 32.7636,
        'lon': 12.7275,
        'elevation': 8,
        'distance_to_coast': 0.0,
        'region': 'northwest'
    },
    'Ajdabiya': {
        'lat': 30.7536,
        'lon': 20.9333,
        'elevation': 50,
        'distance_to_coast': 20.0,
        'region': 'northeast'
    },
    'Sabha': {
        'lat': 27.0167,
        'lon': 14.4333,
        'elevation': 250,
        'distance_to_coast': 500.0,
        'region': 'south'
    },
    'Ghat': {
        'lat': 24.9667,
        'lon': 10.1667,
        'elevation': 500,
        'distance_to_coast': 800.0,
        'region': 'southwest'
    },
    'Ubari': {
        'lat': 26.5833,
        'lon': 12.8167,
        'elevation': 350,
        'distance_to_coast': 600.0,
        'region': 'south'
    },
    'Murzuq': {
        'lat': 25.9167,
        'lon': 13.9167,
        'elevation': 300,
        'distance_to_coast': 700.0,
        'region': 'south'
    }
}

def get_city_geolocation(city):
    """Get geolocation data for a city."""
    return CITY_GEOLOCATION.get(city, {
        'lat': 0.0,
        'lon': 0.0,
        'elevation': 0,
        'distance_to_coast': 0.0,
        'region': 'unknown'
    })
