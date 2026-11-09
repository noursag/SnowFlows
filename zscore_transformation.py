import pandas as pd
from sklearn.preprocessing import StandardScaler

# File paths
input_path = "/Users/reinataher/Desktop/ESIB/Semester 4 Eng./FYP_1/DataSet_before_normalization.csv"
output_path = "/Users/reinataher/Desktop/ESIB/Semester 4 Eng./FYP_1/DataSet_after_zscore.csv"

# Load dataset (two header rows)
df = pd.read_csv(input_path, header=[0, 1])

# Merge headers
df.columns = [
    f"{str(col[0]).strip()}_{str(col[1]).strip()}" if "Unnamed" not in str(col[0])
    else str(col[1]).strip()
    for col in df.columns
]

# Extract week_start column
date_column = df.iloc[:, 0]

# Select numeric feature columns
numeric_df = df.iloc[:, 1:]

# Remove commas from Excel numbers
numeric_df = numeric_df.replace(',', '', regex=True).astype(float)

# Apply Z-score normalization
scaler = StandardScaler()
scaled_values = scaler.fit_transform(numeric_df)

# Convert back to dataframe
df_scaled = pd.DataFrame(scaled_values)

# Rename columns EXACTLY as required
df_scaled.columns = [
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

# Restore week_start column
df_scaled.insert(0, "week_start", date_column)

# Save dataset
df_scaled.to_csv(output_path, index=False)

print("Dataset normalized and renamed successfully.")
print("Saved as DataSet_after_zscore.csv")