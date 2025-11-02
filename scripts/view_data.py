import pandas as pd

# Full path to the dataset
file_path = r"C:\Users\Haneesh.B\Downloads\Solar_Flare_new\scripts\historical_goes_2010_2025_cleaned.csv"

# Load the dataset
df = pd.read_csv(file_path)

# View basic information
print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())
