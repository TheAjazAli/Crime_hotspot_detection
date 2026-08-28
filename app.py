"""
Crime Hotspot Prediction & Visualization Dashboard
----------------------------------------------------
Micro project supporting the research paper:
"An Intelligent Crime Analysis and Hotspot Prediction System Using
Ensemble Learning and GIS-Based Visualization"

Run with:  streamlit run app.py
"""

import folium
from folium.plugins import HeatMap
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from ml_pipeline import (
    DatasetValidationError,
    engineer_features,
    train_and_compare,
    validate_dataset,
)
import analytics
import column_mapping

st.set_page_config(page_title="Crime Hotspot Dashboard", layout="wide")

st.title("Crime Hotspot Prediction & Visualization Dashboard")
st.caption(
    "Upload a crime dataset for a city/region, compare Decision Tree, Random Forest, "
    "XGBoost, and Ensemble models on hotspot prediction, and visualize results on a map."
)

# ---------------------------------------------------------------- Sidebar
st.sidebar.header("1. Data")
uploaded_file = st.sidebar.file_uploader("Upload crime dataset (CSV)", type=["csv"])
use_sample = st.sidebar.checkbox("Use bundled sample dataset instead", value=uploaded_file is None)

st.sidebar.header("2. Hotspot definition")
grid_precision = st.sidebar.slider(
    "Grid cell size (degrees, ~0.01 ≈ 1.1 km)", 0.002, 0.05, 0.01, 0.002
)
hotspot_quantile = st.sidebar.slider(
    "Hotspot threshold (top % of cells by crime count)", 0.5, 0.95, 0.75, 0.05
)

st.sidebar.header("3. Column requirements")
st.sidebar.markdown(
    "Any column names work — you'll map them below after upload.\n\n"
    "- **Required:** latitude, longitude\n"
    "- **Optional (improve accuracy):** crime type, date, hour, weapon, "
    "victim age, status"
)

run_button = st.sidebar.button("Run analysis", type="primary")

# ---------------------------------------------------------------- Load data
df = None
if uploaded_file is not None and not use_sample:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    df = pd.read_csv("sample_crime_data.csv")

if df is None:
    st.info("Upload a CSV in the sidebar, or check 'Use bundled sample dataset' to try it out.")
    st.stop()

with st.expander("Preview uploaded data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"{len(df):,} rows, {len(df.columns)} columns")

# ---------------------------------------------------------------- Column mapping
# Real datasets rarely use our exact column names (e.g. a Kaggle NYPD
# arrests file has `Latitude`, `Arrest_Date`, `Description` instead of
# `latitude`, `date`, `crime_type`). Auto-detect likely matches and let
# the user confirm/override before anything else runs.
st.header("Column mapping")
st.caption(
    "Match your dataset's columns to the fields the app needs. Common variants "
    "(e.g. `Latitude`, `Arrest_Date`, `Description`) are auto-detected below — "
    "just confirm or adjust."
)

detected = column_mapping.auto_detect(list(df.columns))
column_options = ["(none)"] + list(df.columns)
final_mapping = {}

map_col1, map_col2 = st.columns(2)
with map_col1:
    st.markdown("**Required**")
    for field in column_mapping.REQUIRED_FIELDS:
        default = detected.get(field)
        default_idx = column_options.index(default) if default in column_options else 0
        choice = st.selectbox(f"`{field}`", column_options, index=default_idx, key=f"map_{field}")
        final_mapping[field] = None if choice == "(none)" else choice

with map_col2:
    st.markdown("**Optional** (improves prediction accuracy)")
    for field in column_mapping.OPTIONAL_FIELDS:
        default = detected.get(field)
        default_idx = column_options.index(default) if default in column_options else 0
        choice = st.selectbox(f"`{field}`", column_options, index=default_idx, key=f"map_{field}")
        final_mapping[field] = None if choice == "(none)" else choice

missing_required = [f for f in column_mapping.REQUIRED_FIELDS if not final_mapping[f]]
if missing_required:
    st.error(
        f"Please map the required field(s): {', '.join(missing_required)}. "
        f"These weren't auto-detected in your file — pick the matching column above."
    )
    st.stop()

df = column_mapping.apply_mapping(df, final_mapping)

try:
    warnings = validate_dataset(df)
    for w in warnings:
        st.warning(w)
except DatasetValidationError as e:
    st.error(str(e))
    st.stop()

# ---------------------------------------------------------------- Crime data overview
# Descriptive analytics — shown immediately on upload, independent of the ML
# models below. This gives the "which crimes are high" and "what's the
# overall crime picture" context needed to interpret the model results.
st.header("Crime data overview")

stats = analytics.summary_stats(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total records", f"{stats['total_records']:,}")
c2.metric("Crime types", stats["unique_crime_types"] if stats["unique_crime_types"] else "—")
c3.metric("Most common crime", stats["top_crime_type"] or "—")
c4.metric(
    "Avg. crimes / day",
    f"{stats['avg_per_day']:,}" if stats["avg_per_day"] is not None else "—",
)
if stats["date_range"]:
    st.caption(f"Data spans {stats['date_range'][0]} to {stats['date_range'][1]}.")

row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    fig = analytics.crime_type_distribution(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add a `crime_type` column to see which crimes are most frequent.")
with row1_col2:
    fig = analytics.monthly_trend(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add a `date` column to see the crime trend over time.")

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    fig = analytics.hourly_distribution(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
with row2_col2:
    fig = analytics.day_of_week_distribution(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

row3_col1, row3_col2, row3_col3 = st.columns(3)
with row3_col1:
    fig = analytics.weapon_breakdown(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
with row3_col2:
    fig = analytics.status_breakdown(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
with row3_col3:
    fig = analytics.victim_age_histogram(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------- Run pipeline
# IMPORTANT: results are cached in st.session_state, keyed by the dataset +
# settings used to produce them. Streamlit reruns the entire script on every
# interaction — including scrolling/panning the map widgets below — so
# without this cache, any map interaction would wipe the analysis and force
# you to click "Run analysis" again. Caching means the pipeline only
# re-trains when you actually click the button (or change settings and
# click it again); scrolling/panning just redraws from the cached result.
data_signature = (
    uploaded_file.name if (uploaded_file is not None and not use_sample) else "sample_data",
    len(df),
    round(grid_precision, 5),
    round(hotspot_quantile, 5),
    tuple(sorted(final_mapping.items())),
)

if run_button:
    with st.spinner("Engineering features and labeling hotspots..."):
        feat_df, meta = engineer_features(df, grid_precision=grid_precision,
                                           hotspot_quantile=hotspot_quantile)
    with st.spinner("Training Decision Tree, Random Forest, XGBoost, and Ensemble models..."):
        try:
            outcome = train_and_compare(feat_df, meta)
        except DatasetValidationError as e:
            st.error(str(e))
            st.stop()
    st.session_state["signature"] = data_signature
    st.session_state["feat_df"] = feat_df
    st.session_state["meta"] = meta
    st.session_state["outcome"] = outcome

if "outcome" not in st.session_state:
    st.info("Adjust settings in the sidebar, then click **Run analysis**.")
    st.stop()

if st.session_state["signature"] != data_signature:
    st.warning(
        "Settings or dataset changed since the last run — showing results from the "
        "previous run below. Click **Run analysis** again to refresh."
    )

feat_df = st.session_state["feat_df"]
meta = st.session_state["meta"]
outcome = st.session_state["outcome"]

st.success(
    f"Labeled {feat_df['hotspot'].sum():,} of {len(feat_df):,} records as hotspot "
    f"(cells with ≥{meta['threshold_count']:.0f} crimes, using grid size {meta['grid_precision']}°)."
)

results = outcome["results"]
best_name = outcome["best_model_name"]
df_pred = outcome["df_predictions"]

# ---------------------------------------------------------------- Model comparison
st.header("Model comparison")
st.markdown(f"**Best performing model (by F1 score): `{best_name}`**")

metrics_df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "roc_auc"]]
metrics_df = metrics_df.round(3)
st.dataframe(metrics_df, use_container_width=True)

fig = go.Figure()
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    fig.add_trace(go.Bar(name=metric, x=metrics_df.index, y=metrics_df[metric]))
fig.update_layout(
    barmode="group",
    title="Model performance comparison",
    yaxis_title="Score",
    legend_title="Metric",
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

with st.expander(f"Confusion matrix — {best_name}"):
    cm = results[best_name]["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["Actual: Non-hotspot", "Actual: Hotspot"],
                          columns=["Predicted: Non-hotspot", "Predicted: Hotspot"])
    st.dataframe(cm_df, use_container_width=True)

# ---------------------------------------------------------------- Top hotspot locations
st.subheader("Highest-risk locations")
top_cells = analytics.top_hotspot_cells(df_pred, top_n=10)
if not top_cells.empty:
    st.dataframe(top_cells, use_container_width=True)
    st.caption("Grid cells ranked by total crime count — the concrete 'worst areas' in this dataset.")
else:
    st.info("No grid-cell data available for ranking.")

# ---------------------------------------------------------------- Map visualization
st.header("Geographic visualization")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Actual crime density heatmap")
    center = [df_pred["latitude"].mean(), df_pred["longitude"].mean()]
    heat_map = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    HeatMap(df_pred[["latitude", "longitude"]].values.tolist(), radius=10).add_to(heat_map)
    st_folium(heat_map, width=None, height=420, key="heatmap", returned_objects=[])

with col2:
    st.subheader(f"Predicted hotspots ({best_name})")
    pred_map = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    sample_for_map = df_pred.sample(min(1500, len(df_pred)), random_state=1)
    for _, row in sample_for_map.iterrows():
        color = "#d33" if row["predicted_hotspot"] == 1 else "#2a7"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.6,
            weight=0,
        ).add_to(pred_map)
    st_folium(pred_map, width=None, height=420, key="predmap", returned_objects=[])

st.caption("Red = predicted hotspot, green = predicted non-hotspot. "
           "(Up to 1,500 points sampled for map performance.)")

# ---------------------------------------------------------------- Download results
st.header("Export results")
csv_out = df_pred.drop(columns=["grid_lat", "grid_lon", "cell_id"], errors="ignore").to_csv(index=False)
st.download_button(
    "Download predictions as CSV",
    data=csv_out,
    file_name="crime_hotspot_predictions.csv",
    mime="text/csv",
)
metrics_csv = metrics_df.to_csv()
st.download_button(
    "Download model comparison metrics as CSV",
    data=metrics_csv,
    file_name="model_comparison_metrics.csv",
    mime="text/csv",
)