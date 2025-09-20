import pandas as pd

# Full path to the dataset
file_path = r"C:\Users\Haneesh.B\Downloads\Solar_Flare_new\scripts\nasa_flare_events_2010_2025.csv"

# Load the dataset
df = pd.read_csv(file_path)

print("\n Dataset Info (Summary)")
print(df.info())

print("\n Statistical Summary (Numeric Columns)")
print(df.describe())

print("\n Missing Values per Column")
print(df.isnull().sum())
