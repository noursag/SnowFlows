import pandas as pd

# File paths
input_path = "/Users/reinataher/Desktop/ESIB/Semester 4 Eng./FYP_1/weekly_snow_model_ready_fixed_dates.csv"
output_path = "/Users/reinataher/Desktop/ESIB/Semester 4 Eng./FYP_1/weekly_snow_model_ready_with_target.csv"

# Load dataset
df = pd.read_csv(input_path)

# Create target column (shift snow column upward by 1 row)
df["target_snow_next_week"] = df["s2_snow_above_1200_smoothed_zscore"].shift(-1)

# Remove last row (it has no next-week value)
df = df[:-1]

# Save updated dataset
df.to_csv(output_path, index=False)

print("Target column created successfully.")