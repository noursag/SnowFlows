import pandas as pd

file_path = "/Users/reinataher/Desktop/ESIB/Semester 4 Eng./FYP_1/weekly_snow_model_ready_with_target.csv"

df = pd.read_csv(file_path)

# Create split column
df["split"] = ""

# Define split ratios
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

n = len(df)

train_end = int(n * train_ratio)
val_end = int(n * (train_ratio + val_ratio))

df.loc[:train_end, "split"] = "train"
df.loc[train_end:val_end, "split"] = "validation"
df.loc[val_end:, "split"] = "test"

# Move split column to 3rd position
split_col = df.pop("split")
df.insert(2, "split", split_col)

# Save dataset
df.to_csv(file_path, index=False)

print("Split column inserted as the 3rd column successfully.")