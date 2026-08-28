"""
Generates a synthetic crime dataset for a fictional city, so the
dashboard can be tested end-to-end before a real dataset (NCRB,
Kaggle, Chicago/LA/UK police data, etc.) is plugged in.

Run: python generate_sample_data.py
Output: sample_crime_data.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 4000

# Fictional city centered roughly on a lat/lon pair, with a few
# artificial "hotspot" clusters so the model has real signal to learn.
CITY_CENTER = (18.5204, 73.8567)  # Pune-ish, for realism

hotspot_centers = [
    (18.5300, 73.8600, 0.55),   # (lat, lon, weight)
    (18.5100, 73.8450, 0.25),
    (18.5400, 73.8750, 0.20),
]

crime_types = ["Theft", "Assault", "Burglary", "Robbery", "Vandalism",
               "Fraud", "Drug Offense", "Vehicle Theft"]
weapons = ["None", "Knife", "Firearm", "Blunt Object", "Unknown"]
statuses = ["Open", "Closed", "Under Investigation"]

rows = []
for _ in range(N):
    # 70% of records cluster near a hotspot center, 30% scattered randomly
    if np.random.rand() < 0.7:
        center = hotspot_centers[np.random.choice(len(hotspot_centers),
                                                    p=[w for *_, w in hotspot_centers])]
        lat = np.random.normal(center[0], 0.006)
        lon = np.random.normal(center[1], 0.006)
    else:
        lat = np.random.normal(CITY_CENTER[0], 0.03)
        lon = np.random.normal(CITY_CENTER[1], 0.03)

    date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(np.random.rand() * 730))
    hour = int(np.clip(np.random.normal(19, 5), 0, 23))  # more crime in evenings

    rows.append({
        "date": date.strftime("%Y-%m-%d"),
        "hour": hour,
        "crime_type": np.random.choice(crime_types, p=[0.28, 0.15, 0.12, 0.08, 0.12, 0.1, 0.08, 0.07]),
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "city": "Sample City",
        "weapon": np.random.choice(weapons, p=[0.55, 0.15, 0.08, 0.12, 0.10]),
        "victim_age": int(np.clip(np.random.normal(32, 12), 5, 85)),
        "status": np.random.choice(statuses, p=[0.35, 0.5, 0.15]),
    })

df = pd.DataFrame(rows)
df.to_csv("sample_crime_data.csv", index=False)
print(f"Wrote sample_crime_data.csv with {len(df)} rows")
print(df.head())
