import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------
# FILE
# ---------------------------
base_path = Path(__file__).resolve().parent
csv_file = base_path / "MODIS et SENTI2_20162025.csv"

# ---------------------------
# LOAD CSV
# ---------------------------
df = pd.read_csv(csv_file)

df["week_start"] = pd.to_datetime(df["week_start"], dayfirst=True, errors="coerce")
df["snow_km2 MODIS"] = pd.to_numeric(df["snow_km2 MODIS"].astype(str).str.replace(",", ""), errors="coerce")
df["snow_km2 SENTI2"] = pd.to_numeric(df["snow_km2 SENTI2"].astype(str).str.replace(",", ""), errors="coerce")

# ---------------------------
# PLOT
# ---------------------------
fig = go.Figure()

# MODIS (left axis)
fig.add_trace(go.Scatter(
    x=df["week_start"],
    y=df["snow_km2 MODIS"],
    mode="lines",
    name="MODIS",
    hovertemplate="Week: %{x|%Y-%m-%d}<br>MODIS: %{y:.2f} km²<extra></extra>"
))

# SENTI2 (right axis)
fig.add_trace(go.Scatter(
    x=df["week_start"],
    y=df["snow_km2 SENTI2"],
    mode="lines",
    name="SENTI2",
    yaxis="y2",
    hovertemplate="Week: %{x|%Y-%m-%d}<br>SENTI2: %{y:.2f} km²<extra></extra>"
))

# Layout with dual y-axis
fig.update_layout(
    title="Weekly Snow Area: MODIS vs SENTI2",
    xaxis_title="Week Start",
    yaxis=dict(title="MODIS snow area (km²)"),
    yaxis2=dict(title="SENTI2 snow area (km²)", overlaying='y', side='right'),
    hovermode="x unified",
    template="plotly_white"
)

# Save interactive HTML
html_output = base_path / "MODIS_vs_SENTI2_snow_graph.html"
fig.write_html(html_output)
print("Interactive graph saved to:", html_output)

fig.show()