import pandas as pd
import numpy as np

file_path = "/Users/reinataher/Desktop/ESIB/Semester 4 Eng./FYP_1/weekly_snow_model_ready_with_target.csv"
df = pd.read_csv(file_path)

# ----------------------------
# Cyclical encoding
# ----------------------------
df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

# ----------------------------
# Snow persistence
# ----------------------------
df["snow_last_week"] = df["s2_snow_above_1200_smoothed_zscore"].shift(1)
df["snow_last_2weeks"] = df["s2_snow_above_1200_smoothed_zscore"].shift(2)
df["snow_change_ratio"] = df["snow_last_week"] - df["snow_last_2weeks"]

# ----------------------------
# Snow aggregate
# ----------------------------
df["snow_mean"] = (
    df["s2_snow_above_1200_smoothed_zscore"]
    + df["s2_snow_1200_1800_smoothed_zscore"]
    + df["s2_snow_above_1800_smoothed_zscore"]
) / 3

# ----------------------------
# Dynamics
# ----------------------------
df["snow_gradient"] = df["snow_mean"].diff().fillna(0)
df["snow_acceleration"] = df["snow_gradient"].diff().fillna(0)


# ----------------------------

df = df.dropna().reset_index(drop=True)

feature_columns = [
    "week_sin",
    "week_cos",

    "snow_last_week",
    "snow_last_2weeks",
    "snow_change_ratio",
    "snow_mean",
    "snow_gradient",
    "snow_acceleration",

    "s2_snow_above_1200_smoothed_zscore",
    "s2_snow_1200_1800_smoothed_zscore",
    "s2_snow_above_1800_smoothed_zscore",

    "modis_snow_above_1200_smoothed_zscore",
    "modis_snow_1200_1800_smoothed_zscore",
    "modis_snow_above_1800_smoothed_zscore",

    "temp_1200_smoothed_zscore",
    "temp_1200_1800_smoothed_zscore",
    "temp_1800_smoothed_zscore",

    "precip_1200_smoothed_zscore",
    "precip_1200_1800_smoothed_zscore",
    "precip_1800_smoothed_zscore"
]

target_column = "target_snow_next_week"

sequence_length = 8 ## sweet spot is 8 weeks after several trials

X = []
y = []

for i in range(len(df) - sequence_length):
    X.append(df[feature_columns].iloc[i:i + sequence_length].values)
    y.append(df[target_column].iloc[i + sequence_length])

X = np.array(X)
y = np.array(y)

split_sequence = df["split"].iloc[sequence_length-1:-1].values

train_mask = split_sequence == "train"
val_mask = split_sequence == "validation"
test_mask = split_sequence == "test"

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Sequences saved successfully")
print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)
print("Number of features used:", len(feature_columns))