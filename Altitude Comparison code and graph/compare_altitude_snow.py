import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# -------------------------------------------------
# FILES
# Put the 6 CSV files in the same folder as this script
# -------------------------------------------------
files = {
    "Above 1200 m": [
        "Sentinel2_Weekly_Snow_above1200m_2016_2021.csv",
        "Sentinel2_Weekly_Snow_above1200m_2021_2025.csv",
    ],
    "1200–1800 m": [
        "Sentinel2_Weekly_Snow_1200_1800m_2016_2021.csv",
        "Sentinel2_Weekly_Snow_1200_1800m_2021_2025.csv",
    ],
    "Above 1800 m": [
        "Sentinel2_Weekly_Snow_above1800m_2016_2021.csv",
        "Sentinel2_Weekly_Snow_above1800m_2021_2025.csv",
    ],
}

base_path = Path(__file__).resolve().parent

# -------------------------------------------------
# LOAD + CLEAN ONE ALTITUDE CLASS
# -------------------------------------------------
def load_altitude_class(file_list, altitude_label):
    dfs = []

    for file_name in file_list:
        file_path = base_path / file_name
        df = pd.read_csv(file_path)

        # Keep only useful columns
        keep_cols = ["week_start", "week_end", "snow_km2", "snow_pct"]
        df = df[[col for col in keep_cols if col in df.columns]].copy()

        # Convert dates
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
        if "week_end" in df.columns:
            df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")

        # Convert numeric columns
        df["snow_km2"] = pd.to_numeric(df["snow_km2"], errors="coerce")
        if "snow_pct" in df.columns:
            df["snow_pct"] = pd.to_numeric(df["snow_pct"], errors="coerce")

        # Add altitude label
        df["altitude_class"] = altitude_label

        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    # Remove bad rows
    merged = merged.dropna(subset=["week_start", "snow_km2"])

    # Sort by date
    merged = merged.sort_values("week_start")

    # Remove duplicate boundary dates like 2021-10-01 if repeated
    merged = merged.drop_duplicates(subset=["altitude_class", "week_start"], keep="first")

    return merged

# -------------------------------------------------
# LOAD ALL DATA
# -------------------------------------------------
all_dfs = []
for altitude_label, file_list in files.items():
    all_dfs.append(load_altitude_class(file_list, altitude_label))

df_all = pd.concat(all_dfs, ignore_index=True)
df_all = df_all.sort_values(["altitude_class", "week_start"])

# Optional: save merged clean dataset
merged_output = base_path / "merged_altitude_snow_dataset.csv"
df_all.to_csv(merged_output, index=False)

print("Merged dataset saved to:", merged_output)
print(df_all.head())

# -------------------------------------------------
# INTERACTIVE GRAPH
# -------------------------------------------------
fig = go.Figure()

for altitude_label in ["Above 1200 m", "1200–1800 m", "Above 1800 m"]:
    d = df_all[df_all["altitude_class"] == altitude_label]

    fig.add_trace(
        go.Scatter(
            x=d["week_start"],
            y=d["snow_km2"],
            mode="lines",
            name=altitude_label,
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Week: %{x|%Y-%m-%d}<br>"
                "Snow area: %{y:.2f} km²"
                "<extra></extra>"
            ),
        )
    )

fig.update_layout(
    title="Sentinel-2 Weekly Snow Area by Altitude (2016-10-01 to 2025-06-01)",
    xaxis_title="Week start",
    yaxis_title="Snow area (km²)",
    hovermode="x unified",
    template="plotly_white",
    legend_title="Altitude class",
)

# Save interactive HTML graph
html_output = base_path / "interactive_altitude_snow_graph.html"
fig.write_html(html_output)

print("Interactive graph saved to:", html_output)

# Show graph
fig.show()