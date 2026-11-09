import pandas as pd
from pathlib import Path

# =========================================================
# SETTINGS
# =========================================================

INPUT_FILE = "Lebanon_Weekly_MODIS_25years_3altitudes.csv.xlsx"
OUTPUT_FILE = "modis_25years_ready_with_target.csv"

SMOOTHING_WINDOW = 2  # every 2 consecutive values

MODIS_COLUMNS = [
    "modis_snow_above_1200",
    "modis_snow_1200_1800",
    "modis_snow_above_1800"
]

# =========================================================
# LOAD FILE
# =========================================================

input_path = Path(INPUT_FILE)

if not input_path.exists():
    raise FileNotFoundError(f"File not found: {INPUT_FILE}")

if input_path.suffix.lower() == ".csv":
    df = pd.read_csv(input_path)
elif input_path.suffix.lower() in [".xlsx", ".xls"]:
    df = pd.read_excel(input_path)
else:
    raise ValueError("Input file must be .csv, .xlsx, or .xls")

# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_columns = ["week_start"] + MODIS_COLUMNS
missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Keep only useful columns
df = df[required_columns].copy()

# =========================================================
# DATE PROCESSING
# =========================================================

df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
df = df.dropna(subset=["week_start"]).copy()
df = df.sort_values("week_start").reset_index(drop=True)

iso = df["week_start"].dt.isocalendar()
df["year"] = iso.year.astype(int)
df["week_of_year"] = iso.week.astype(int)

# =========================================================
# CLEAN NUMERIC COLUMNS
# =========================================================

for col in MODIS_COLUMNS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=MODIS_COLUMNS).reset_index(drop=True)

# =========================================================
# Z-SCORE NORMALIZATION
# =========================================================

for col in MODIS_COLUMNS:
    mean_val = df[col].mean()
    std_val = df[col].std()

    z_col = f"{col}_zscore"

    if std_val == 0 or pd.isna(std_val):
        df[z_col] = 0.0
    else:
        df[z_col] = (df[col] - mean_val) / std_val

# =========================================================
# SMOOTHING (WINDOW = 2)
# =========================================================

for col in MODIS_COLUMNS:
    z_col = f"{col}_zscore"
    sm_col = f"{col}_smoothed_zscore"

    df[sm_col] = (
        df[z_col]
        .rolling(window=SMOOTHING_WINDOW, min_periods=1)
        .mean()
    )

# =========================================================
# TARGET COLUMN
# predict next week's MODIS snow above 1200
# =========================================================

df["target_modis_next_week"] = df["modis_snow_above_1200_smoothed_zscore"].shift(-1)

# remove final row with no target
df = df.dropna(subset=["target_modis_next_week"]).reset_index(drop=True)

# =========================================================
# SPLIT: 70 / 15 / 15
# =========================================================

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

df["split"] = "test"
df.loc[:train_end - 1, "split"] = "train"
df.loc[train_end:val_end - 1, "split"] = "validation"
df.loc[val_end:, "split"] = "test"

# =========================================================
# FINAL COLUMN ORDER
# =========================================================

final_columns = [
    "year",
    "week_of_year",
    "week_start",

    "modis_snow_above_1200",
    "modis_snow_1200_1800",
    "modis_snow_above_1800",

    "modis_snow_above_1200_zscore",
    "modis_snow_1200_1800_zscore",
    "modis_snow_above_1800_zscore",

    "modis_snow_above_1200_smoothed_zscore",
    "modis_snow_1200_1800_smoothed_zscore",
    "modis_snow_above_1800_smoothed_zscore",

    "target_modis_next_week",
    "split"
]

df = df[final_columns]

# =========================================================
# SAVE
# =========================================================

df.to_csv(OUTPUT_FILE, index=False)

print("Done.")
print(f"Saved file: {OUTPUT_FILE}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print("\nColumns:")
print(df.columns.tolist())
print("\nHead:")
print(df.head())