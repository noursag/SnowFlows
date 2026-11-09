import pandas as pd
from pathlib import Path

# =========================================================
# FILES
# =========================================================

RAW_INPUT_FILE = "Lebanon_Weekly_MODIS_25years_3altitudes from 2000 till 2025.xlsx"
READY_INPUT_FILE = "modis_25years_ready_with_target.csv"
OUTPUT_FILE = "modis_25years_ready_with_target_full.csv"

SMOOTHING_WINDOW = 2  # 2 consecutive values

# =========================================================
# LOAD
# =========================================================

raw_path = Path(RAW_INPUT_FILE)
ready_path = Path(READY_INPUT_FILE)

if not raw_path.exists():
    raise FileNotFoundError(f"Raw Excel file not found: {RAW_INPUT_FILE}")

if not ready_path.exists():
    raise FileNotFoundError(f"Ready CSV file not found: {READY_INPUT_FILE}")

# Important: your real header is on row 2
raw_df = pd.read_excel(raw_path, header=1)
ready_df = pd.read_csv(ready_path)

# =========================================================
# RENAME RAW EXCEL COLUMNS TO MATCH YOUR READY STYLE
# =========================================================

raw_df = raw_df.rename(columns={
    "precipitation_mm_above_1200": "precip_1200",
    "precipitation_mm_between_1200-1800": "precip_1200_1800",
    "precipitation_mm_above_1800": "precip_1800",
    "temperature_c_above_1200": "temp_1200",
    "temperature_c_between_1200-1800": "temp_1200_1800",
    "temperature_c_above_1800": "temp_1800",
    "modis_snow_between_1200-1800": "modis_snow_1200_1800",
})

# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_raw = [
    "week_start",
    "temp_1200",
    "temp_1200_1800",
    "temp_1800",
    "precip_1200",
    "precip_1200_1800",
    "precip_1800",
]

missing = [c for c in required_raw if c not in raw_df.columns]
if missing:
    raise ValueError(f"Missing columns in raw Excel after renaming: {missing}")

required_ready = [
    "year",
    "week_of_year",
    "week_start",
    "target_modis_next_week",
    "split",
]

missing_ready = [c for c in required_ready if c not in ready_df.columns]
if missing_ready:
    raise ValueError(f"Missing columns in ready CSV: {missing_ready}")

# =========================================================
# KEEP ONLY USEFUL RAW COLUMNS
# =========================================================

raw_df = raw_df[required_raw].copy()

# =========================================================
# DATE CLEANING
# =========================================================

raw_df["week_start"] = pd.to_datetime(raw_df["week_start"], errors="coerce")
ready_df["week_start"] = pd.to_datetime(ready_df["week_start"], errors="coerce")

raw_df = raw_df.dropna(subset=["week_start"]).copy()
ready_df = ready_df.dropna(subset=["week_start"]).copy()

raw_df = raw_df.sort_values("week_start").drop_duplicates(subset=["week_start"]).reset_index(drop=True)
ready_df = ready_df.sort_values("week_start").drop_duplicates(subset=["week_start"]).reset_index(drop=True)

# =========================================================
# CHECK CONSECUTIVE WEEKS
# =========================================================

week_diff = raw_df["week_start"].diff().dt.days
bad_gaps = raw_df.loc[(week_diff.notna()) & (week_diff != 7), ["week_start"]]

if not bad_gaps.empty:
    print("Warning: some rows are not exactly 7 days apart.")
    print(bad_gaps.head(10))

# =========================================================
# NUMERIC CLEANING
# =========================================================

temp_precip_cols = [
    "temp_1200",
    "temp_1200_1800",
    "temp_1800",
    "precip_1200",
    "precip_1200_1800",
    "precip_1800",
]

for col in temp_precip_cols:
    raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

raw_df = raw_df.dropna(subset=temp_precip_cols).reset_index(drop=True)

# =========================================================
# NORMALIZE ONLY TEMP + PRECIP
# =========================================================

for col in temp_precip_cols:
    mean_val = raw_df[col].mean()
    std_val = raw_df[col].std()

    if pd.isna(std_val) or std_val == 0:
        raise ValueError(f"Column {col} has invalid/zero std.")

    raw_df[f"{col}_zscore"] = (raw_df[col] - mean_val) / std_val

# =========================================================
# SMOOTH WITH 2 CONSECUTIVE VALUES
# =========================================================

for col in temp_precip_cols:
    z_col = f"{col}_zscore"
    sm_col = f"{col}_smoothed_zscore"

    raw_df[sm_col] = (
        raw_df[z_col]
        .rolling(window=SMOOTHING_WINDOW, min_periods=1)
        .mean()
    )

# =========================================================
# KEEP ONLY NEW COLUMNS TO ADD
# =========================================================

new_cols_df = raw_df[
    [
        "week_start",

        "temp_1200_zscore",
        "temp_1200_1800_zscore",
        "temp_1800_zscore",
        "temp_1200_smoothed_zscore",
        "temp_1200_1800_smoothed_zscore",
        "temp_1800_smoothed_zscore",

        "precip_1200_zscore",
        "precip_1200_1800_zscore",
        "precip_1800_zscore",
        "precip_1200_smoothed_zscore",
        "precip_1200_1800_smoothed_zscore",
        "precip_1800_smoothed_zscore",
    ]
].copy()

# =========================================================
# REMOVE OLD TEMP/PRECIP COLUMNS IF THEY ALREADY EXIST
# =========================================================

cols_to_remove = [c for c in new_cols_df.columns if c in ready_df.columns and c != "week_start"]
if cols_to_remove:
    ready_df = ready_df.drop(columns=cols_to_remove)

# =========================================================
# MERGE INTO EXISTING READY CSV
# =========================================================

merged_df = ready_df.merge(new_cols_df, on="week_start", how="left")

# =========================================================
# REORDER: keep original ready columns first
# =========================================================

front_cols = [
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
    "split",

    "temp_1200_zscore",
    "temp_1200_1800_zscore",
    "temp_1800_zscore",
    "temp_1200_smoothed_zscore",
    "temp_1200_1800_smoothed_zscore",
    "temp_1800_smoothed_zscore",

    "precip_1200_zscore",
    "precip_1200_1800_zscore",
    "precip_1800_zscore",
    "precip_1200_smoothed_zscore",
    "precip_1200_1800_smoothed_zscore",
    "precip_1800_smoothed_zscore",
]

front_cols = [c for c in front_cols if c in merged_df.columns]
other_cols = [c for c in merged_df.columns if c not in front_cols]

merged_df = merged_df[front_cols + other_cols]

# =========================================================
# SAVE
# =========================================================

merged_df.to_csv(OUTPUT_FILE, index=False)

print("Done.")
print(f"Saved: {OUTPUT_FILE}")
print(f"Rows: {len(merged_df)}")
print(f"Columns: {len(merged_df.columns)}")
print("\nFinal columns:")
print(merged_df.columns.tolist())