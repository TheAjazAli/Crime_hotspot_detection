"""
Descriptive crime analytics — the "what does the data actually show"
layer, independent of the ML hotspot models. This answers questions like
"which crime types are most common", "when do crimes happen most", and
"what's the overall crime rate", which are needed context for interpreting
the model comparison results.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def summary_stats(df: pd.DataFrame) -> dict:
    """High-level numbers for the metric cards at the top of the dashboard."""
    stats = {
        "total_records": len(df),
        "unique_crime_types": df["crime_type"].nunique() if "crime_type" in df.columns else None,
        "date_range": None,
        "avg_per_day": None,
        "top_crime_type": None,
        "busiest_hour": None,
    }

    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce").dropna()
        if not parsed.empty:
            span_days = max((parsed.max() - parsed.min()).days, 1)
            stats["date_range"] = (parsed.min().date(), parsed.max().date())
            stats["avg_per_day"] = round(len(df) / span_days, 1)

    if "crime_type" in df.columns and df["crime_type"].notna().any():
        stats["top_crime_type"] = df["crime_type"].value_counts().idxmax()

    if "hour" in df.columns:
        hours = pd.to_numeric(df["hour"], errors="coerce").dropna()
        if not hours.empty:
            stats["busiest_hour"] = int(hours.mode().iloc[0])

    return stats


def crime_type_distribution(df: pd.DataFrame, top_n: int = 10):
    """Bar chart ranking crime types by frequency — 'which crimes are high'."""
    if "crime_type" not in df.columns:
        return None
    counts = df["crime_type"].value_counts().head(top_n).reset_index()
    counts.columns = ["crime_type", "count"]
    fig = px.bar(
        counts, x="count", y="crime_type", orientation="h",
        title="Most frequent crime types", labels={"count": "Number of records", "crime_type": ""},
        color="count", color_continuous_scale="Reds",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False, height=380)
    return fig


def monthly_trend(df: pd.DataFrame):
    """Line chart of crime volume over time — shows whether the crime rate is rising/falling."""
    if "date" not in df.columns:
        return None
    parsed = pd.to_datetime(df["date"], errors="coerce").dropna()
    if parsed.empty:
        return None
    monthly = parsed.dt.to_period("M").value_counts().sort_index()
    monthly.index = monthly.index.astype(str)
    fig = px.line(
        x=monthly.index, y=monthly.values, markers=True,
        title="Crime volume over time",
        labels={"x": "Month", "y": "Number of crimes"},
    )
    fig.update_layout(height=350)
    return fig


def hourly_distribution(df: pd.DataFrame):
    """Bar chart of crimes by hour of day."""
    if "hour" not in df.columns:
        return None
    hours = pd.to_numeric(df["hour"], errors="coerce").dropna().astype(int)
    if hours.empty:
        return None
    counts = hours.value_counts().reindex(range(24), fill_value=0).sort_index()
    fig = px.bar(
        x=counts.index, y=counts.values,
        title="Crimes by hour of day",
        labels={"x": "Hour (24h)", "y": "Number of crimes"},
        color=counts.values, color_continuous_scale="Oranges",
    )
    fig.update_layout(coloraxis_showscale=False, height=320)
    return fig


def day_of_week_distribution(df: pd.DataFrame):
    """Bar chart of crimes by day of week."""
    if "date" not in df.columns:
        return None
    parsed = pd.to_datetime(df["date"], errors="coerce").dropna()
    if parsed.empty:
        return None
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    counts = parsed.dt.dayofweek.value_counts().reindex(range(7), fill_value=0)
    counts.index = day_names
    fig = px.bar(
        x=counts.index, y=counts.values,
        title="Crimes by day of week",
        labels={"x": "", "y": "Number of crimes"},
        color=counts.values, color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False, height=320)
    return fig


def weapon_breakdown(df: pd.DataFrame):
    """Pie chart of weapon usage, if the column exists."""
    if "weapon" not in df.columns:
        return None
    counts = df["weapon"].value_counts().reset_index()
    counts.columns = ["weapon", "count"]
    fig = px.pie(counts, names="weapon", values="count", title="Weapon involvement", hole=0.4)
    fig.update_layout(height=360)
    return fig


def status_breakdown(df: pd.DataFrame):
    """Pie chart of case status, if the column exists."""
    if "status" not in df.columns:
        return None
    counts = df["status"].value_counts().reset_index()
    counts.columns = ["status", "count"]
    fig = px.pie(counts, names="status", values="count", title="Case status breakdown", hole=0.4)
    fig.update_layout(height=360)
    return fig


def victim_age_histogram(df: pd.DataFrame):
    """Histogram of victim age, if the column exists."""
    if "victim_age" not in df.columns:
        return None
    ages = pd.to_numeric(df["victim_age"], errors="coerce").dropna()
    if ages.empty:
        return None
    fig = px.histogram(ages, nbins=20, title="Victim age distribution",
                        labels={"value": "Victim age", "count": "Number of records"})
    fig.update_layout(showlegend=False, height=320)
    return fig


def top_hotspot_cells(df_with_cells: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Ranks grid cells by crime count — a concrete 'these are the worst areas'
    table, using the cell_id/grid_lat/grid_lon/cell_crime_count columns
    produced by ml_pipeline.engineer_features.
    """
    required = {"cell_id", "grid_lat", "grid_lon", "cell_crime_count"}
    if not required.issubset(df_with_cells.columns):
        return pd.DataFrame()
    top = (
        df_with_cells[["cell_id", "grid_lat", "grid_lon", "cell_crime_count"]]
        .drop_duplicates("cell_id")
        .sort_values("cell_crime_count", ascending=False)
        .head(top_n)
        .rename(columns={
            "grid_lat": "latitude (cell center)",
            "grid_lon": "longitude (cell center)",
            "cell_crime_count": "crime count",
        })
        .drop(columns=["cell_id"])
        .reset_index(drop=True)
    )
    top.index = top.index + 1
    return top