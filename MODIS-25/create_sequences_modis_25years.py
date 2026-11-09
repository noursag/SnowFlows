import pandas as pd
import numpy as np
import os

# ============================
# AUTO FILE PATH
# ============================

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "modis_25years_ready_with_target_full.csv")

print("Looking for file at:", file_path)

if not os.path.exists(file_path):
    raise FileNotFoundError(f"CSV not found at: {file_path}")

df = pd.read_csv(file_path)

# ============================
# CHECK REQUIRED COLUMNS
# ============================

required_columns = [
    "week_of_year",

    "modis_snow_above_1200_zscore",
    "modis_snow_1200_1800_zscore",
    "modis_snow_above_1800_zscore",

    "temp_1200_smoothed_zscore",
    "temp_1200_1800_smoothed_zscore",
    "temp_1800_smoothed_zscore",

    "precip_1200_smoothed_zscore",
    "precip_1200_1800_smoothed_zscore",
    "precip_1800_smoothed_zscore",

    "target_modis_next_week",
    "split"
]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ============================
# CYCLICAL ENCODING
# ============================

df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

# ============================
# SNOW PERSISTENCE
# ============================

df["snow_last_week"] = df["modis_snow_above_1200_zscore"].shift(1)
df["snow_last_2weeks"] = df["modis_snow_above_1200_zscore"].shift(2)
df["snow_change_ratio"] = df["snow_last_week"] - df["snow_last_2weeks"]

# ============================
# SNOW AGGREGATE
# ============================

df["snow_mean"] = (
    df["modis_snow_above_1200_zscore"]
    + df["modis_snow_1200_1800_zscore"]
    + df["modis_snow_above_1800_zscore"]
) / 3

# ============================
# SNOW DYNAMICS
# ============================

df["snow_gradient"] = df["snow_mean"].diff().fillna(0)
df["snow_acceleration"] = df["snow_gradient"].diff().fillna(0)

# ============================
# CLEAN
# ============================

df = df.dropna().reset_index(drop=True)

# ============================
# FEATURES
# ============================

feature_columns = [
    "week_sin",
    "week_cos",

    "snow_last_week",
    "snow_last_2weeks",
    "snow_change_ratio",
    "snow_mean",
    "snow_gradient",
    "snow_acceleration",

    "modis_snow_above_1200_zscore",
    "modis_snow_1200_1800_zscore",
    "modis_snow_above_1800_zscore",

    "temp_1200_smoothed_zscore",
    "temp_1200_1800_smoothed_zscore",
    "temp_1800_smoothed_zscore",

    "precip_1200_smoothed_zscore",
    "precip_1200_1800_smoothed_zscore",
    "precip_1800_smoothed_zscore"
]

target_column = "target_modis_next_week"

# ============================
# CHANGE THIS ONLY
# ============================

sequence_length = 6

# ============================
# BUILD SEQUENCES
# ============================

X = []
y = []

for i in range(len(df) - sequence_length):
    X.append(df[feature_columns].iloc[i:i + sequence_length].values)
    y.append(df[target_column].iloc[i + sequence_length])

X = np.array(X)
y = np.array(y)

# ============================
# SPLIT
# ============================

split_sequence = df["split"].iloc[sequence_length - 1:-1].values

train_mask = split_sequence == "train"
val_mask = split_sequence == "validation"
test_mask = split_sequence == "test"

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

# ============================
# SAVE
# ============================

np.save("X_train_modis_25_full.npy", X_train)
np.save("y_train_modis_25_full.npy", y_train)
np.save("X_val_modis_25_full.npy", X_val)
np.save("y_val_modis_25_full.npy", y_val)
np.save("X_test_modis_25_full.npy", X_test)
np.save("y_test_modis_25_full.npy", y_test)

# ============================
# DEBUG PRINTS
# ============================

print("\n✅ Sequences saved successfully")
print("Sequence length:", sequence_length)
print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)
print("Number of features:", len(feature_columns))