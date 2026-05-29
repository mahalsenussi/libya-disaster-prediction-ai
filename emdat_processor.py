#!/usr/bin/env python3
"""
EM-DAT disaster data processor.
Extracts Libya disaster events and creates labels for training data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from city_geolocation import CITY_GEOLOCATION
import re

EMDAT_PATH = "/var/www/html/ml/data/public_emdat_incl_hist_2026-05-25.xlsx"

def load_emdat():
    """Load EM-DAT data from Excel file."""
    print(f"Loading EM-DAT data from {EMDAT_PATH}...")
    df = pd.read_excel(EMDAT_PATH)
    print(f"Loaded {len(df)} total disasters worldwide.")
    return df

def extract_libya_disasters(df):
    """Extract Libya-specific disasters from EM-DAT."""
    libya_df = df[df['Country'] == 'Libya'].copy()
    print(f"Found {len(libya_df)} Libya disasters.")
    
    # Filter relevant disaster types for our system
    relevant_types = ['Flood', 'Storm', 'Water', 'Earthquake', 'Landslide']
    libya_df = libya_df[libya_df['Disaster Type'].isin(relevant_types)]
    print(f"Filtered to {len(libya_df)} relevant disasters (Flood, Storm, Water, etc.).")
    
    return libya_df

def parse_location(location_str):
    """Parse location string to extract city names."""
    if pd.isna(location_str):
        return []
    
    location_str = str(location_str).lower()
    
    # List of Libyan cities to match
    cities = []
    for city in CITY_GEOLOCATION.keys():
        if city.lower() in location_str:
            cities.append(city)
    
    return cities if cities else None

def calculate_severity(row):
    """Calculate disaster severity based on deaths and affected people."""
    deaths = row.get('Total Deaths', 0)
    affected = row.get('No. Affected', 0)
    total_affected = row.get('Total Affected', affected)
    
    if pd.isna(deaths):
        deaths = 0
    if pd.isna(total_affected):
        total_affected = 0
    
    # Severity scoring
    if deaths >= 100 or total_affected >= 100000:
        return 3  # SEVERE
    elif deaths >= 10 or total_affected >= 10000:
        return 2  # MODERATE
    elif deaths >= 1 or total_affected >= 1000:
        return 1  # MINOR
    else:
        return 1  # MINOR (default for any disaster)

def process_libya_disasters(libya_df):
    """Process Libya disasters into structured format."""
    processed = []
    
    for idx, row in libya_df.iterrows():
        # Parse date
        start_year = row.get('Start Year')
        start_month = row.get('Start Month')
        start_day = row.get('Start Day')
        
        # Handle NaN values
        if pd.isna(start_year) or start_year == 0:
            continue
        start_year = int(start_year)
        start_month = int(start_month) if pd.notna(start_month) else 1
        start_day = int(start_day) if pd.notna(start_day) else 1
        
        try:
            start_date = datetime(start_year, start_month, start_day)
        except:
            continue
        
        # Parse location
        location_str = row.get('Location', '')
        cities = parse_location(location_str)
        
        # If no city matched, try to use lat/lon to find nearest city
        if not cities:
            lat = row.get('Latitude')
            lon = row.get('Longitude')
            if pd.notna(lat) and pd.notna(lon):
                # Find nearest city (simplified - could use proper distance calculation)
                nearest_city = min(CITY_GEOLOCATION.keys(), 
                                 key=lambda c: (CITY_GEOLOCATION[c]['lat'] - lat)**2 + 
                                               (CITY_GEOLOCATION[c]['lon'] - lon)**2)
                cities = [nearest_city]
        
        # Calculate severity
        severity = calculate_severity(row)
        
        disaster = {
            'disaster_no': row.get('DisNo.', ''),
            'disaster_type': row.get('Disaster Type', ''),
            'disaster_subtype': row.get('Disaster Subtype', ''),
            'start_date': start_date,
            'end_date': None,  # Could parse end date if needed
            'location': location_str,
            'cities': cities,
            'latitude': row.get('Latitude'),
            'longitude': row.get('Longitude'),
            'total_deaths': row.get('Total Deaths', 0),
            'no_affected': row.get('No. Affected', 0),
            'total_affected': row.get('Total Affected', 0),
            'severity': severity
        }
        
        processed.append(disaster)
    
    print(f"Processed {len(processed)} disasters with valid dates and locations.")
    return processed

def save_processed_disasters(disasters, output_path):
    """Save processed disasters to CSV."""
    rows = []
    for d in disasters:
        row = {
            'disaster_no': d['disaster_no'],
            'disaster_type': d['disaster_type'],
            'disaster_subtype': d['disaster_subtype'],
            'start_date': d['start_date'].isoformat(),
            'location': d['location'],
            'cities': ','.join(d['cities']) if d['cities'] else '',
            'latitude': d['latitude'],
            'longitude': d['longitude'],
            'total_deaths': d['total_deaths'],
            'no_affected': d['no_affected'],
            'total_affected': d['total_affected'],
            'severity': d['severity']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved processed disasters to {output_path}")

def main():
    """Main processing function."""
    # Load EM-DAT
    df = load_emdat()
    
    # Extract Libya disasters
    libya_df = extract_libya_disasters(df)
    
    # Process disasters
    processed = process_libya_disasters(libya_df)
    
    # Save processed disasters
    output_path = "/var/www/html/ml/data/libya_disasters.csv"
    save_processed_disasters(processed, output_path)
    
    # Print summary
    print("\n=== Disaster Summary ===")
    by_type = {}
    by_severity = {1: 0, 2: 0, 3: 0}
    for d in processed:
        dtype = d['disaster_type']
        by_type[dtype] = by_type.get(dtype, 0) + 1
        by_severity[d['severity']] += 1
    
    print("\nBy Disaster Type:")
    for dtype, count in sorted(by_type.items()):
        print(f"  {dtype}: {count}")
    
    print("\nBy Severity:")
    print(f"  Minor (1): {by_severity[1]}")
    print(f"  Moderate (2): {by_severity[2]}")
    print(f"  Severe (3): {by_severity[3]}")
    
    print("\nSample disasters:")
    for d in processed[:5]:
        print(f"  {d['start_date'].date()} - {d['disaster_type']} - {d['cities']} - Severity: {d['severity']}")

if __name__ == "__main__":
    main()
