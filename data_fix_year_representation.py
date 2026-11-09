import pandas as pd

# File paths
input_path = "/Users/reinataher/Desktop/ESIB/Semester 4 Eng./FYP_1/DataSet_after_zscore.csv"
output_path = "/Users/reinataher/Desktop/ESIB/Semester 4 Eng./FYP_1/weekly_snow_model_ready_fixed_dates.csv"

# Load dataset
df = pd.read_csv(input_path)

# Convert date column
df["week_start"] = pd.to_datetime(df["week_start"], dayfirst=True)

# Extract year and week number
df["year"] = df["week_start"].dt.year
df["week_of_year"] = df["week_start"].dt.isocalendar().week

# Remove original column
df.drop(columns=["week_start"], inplace=True)

# Move year and week_of_year to the front
cols = ["year", "week_of_year"] + [col for col in df.columns if col not in ["year", "week_of_year"]]
df = df[cols]

# Save file
df.to_csv(output_path, index=False)

print("Date columns reordered successfully.")