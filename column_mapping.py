"""
Column mapping — real-world datasets (Kaggle, NCRB, city open-data portals)
almost never use the exact column names `latitude`, `crime_type`, `date`,
etc. This module auto-detects likely matches from common naming
conventions (e.g. `Latitude`, `Arrest_Date`, `Description`) and lets the
user confirm or override the mapping in the UI, then renames the raw
dataframe to the app's standard schema so the rest of the pipeline
(validate_dataset, engineer_features, ...) doesn't need to change.
"""

from __future__ import annotations

import pandas as pd

# Standard field name -> list of common raw-column aliases to look for
# (matched case-insensitively, with underscores/spaces treated the same).
FIELD_ALIASES = {
    "latitude": ["latitude", "lat", "y_coordinate_wgs84"],
    "longitude": ["longitude", "lon", "lng", "long", "x_coordinate_wgs84"],
    "crime_type": ["crime_type", "description", "offense", "offense_description",
                    "crime", "crm_cd_desc", "primary_type", "law_code"],
    "date": ["date", "arrest_date", "incident_date", "occurred_date", "date_occ",
              "rpt_date", "report_date"],
    "hour": ["hour", "hour_of_day", "time_occ"],
    "weapon": ["weapon", "weapon_used", "weapon_desc"],
    "victim_age": ["victim_age", "vict_age"],
    "status": ["status", "case_status", "indicator", "status_desc"],
}

REQUIRED_FIELDS = ["latitude", "longitude"]
OPTIONAL_FIELDS = ["crime_type", "date", "hour", "weapon", "victim_age", "status"]


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def auto_detect(columns: list[str]) -> dict[str, str | None]:
    """Returns {standard_field: best_guess_raw_column_or_None} for every field."""
    normalized_lookup = {_normalize(c): c for c in columns}
    detected = {}
    for field, aliases in FIELD_ALIASES.items():
        match = None
        for alias in aliases:
            if alias in normalized_lookup:
                match = normalized_lookup[alias]
                break
        detected[field] = match
    return detected


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """
    Returns a new dataframe with columns renamed to the standard schema,
    based on a {standard_field: raw_column_or_None} mapping. Unmapped
    standard fields are simply absent from the output (the rest of the
    pipeline already treats them as optional).
    """
    rename_map = {raw: std for std, raw in mapping.items() if raw}
    out = df.rename(columns=rename_map)
    # Keep only the standardized columns plus anything unmapped/original,
    # so nothing is silently dropped.
    return out