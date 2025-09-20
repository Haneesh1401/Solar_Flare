import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ----------------------------
# Load and preprocess dataset
# ----------------------------
df = pd.read_csv("nasa_flare_events_2010_2025.csv")
df['beginTime'] = pd.to_datetime(df['beginTime'], errors='coerce')
df['endTime'] = pd.to_datetime(df['endTime'], errors='coerce')

# Duration in minutes
df['duration_minutes'] = (df['endTime'] - df['beginTime']).dt.total_seconds() / 60
df = df.dropna(subset=['duration_minutes'])

# Clean dataset: remove invalid durations (negative or > 24 hrs)
df_clean = df[(df['duration_minutes'] > 0) & (df['duration_minutes'] < 1440)]

# ----------------------------
# 1. Bar Chart - Flare Class Frequency
# ----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='classType', data=df_clean, order=df_clean['classType'].value_counts().index, palette="viridis")
plt.title("Frequency of Solar Flare Classes")
plt.xlabel("Flare Class")
plt.ylabel("Count")
plt.show()

# ----------------------------
# 2. Histogram - Flare Duration Distribution
# ----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df_clean['duration_minutes'], bins=30, kde=True, color="coral", edgecolor="black")

# Add mean & median lines
plt.axvline(df_clean['duration_minutes'].mean(), color='red', linestyle='--', label=f"Mean: {df_clean['duration_minutes'].mean():.1f} min")
plt.axvline(df_clean['duration_minutes'].median(), color='green', linestyle='-', label=f"Median: {df_clean['duration_minutes'].median():.1f} min")

plt.title("Distribution of Flare Durations (Cleaned)")
plt.xlabel("Duration (minutes)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# ----------------------------
# 3. Boxplot - Flare Duration by Class
# ----------------------------
plt.figure(figsize=(12,6))
sns.boxplot(x='classType', y='duration_minutes', data=df_clean, palette="Set2")
plt.title("Boxplot of Flare Duration by Class Type (Cleaned)")
plt.xlabel("Flare Class")
plt.ylabel("Duration (minutes)")
plt.xticks(rotation=90)
plt.show()

# ----------------------------
# 4. Pie Chart - Flare Class Proportions
# ----------------------------
plt.figure(figsize=(7,7))
df_clean['classType'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='Set3')
plt.title("Proportion of Solar Flare Classes")
plt.ylabel("")
plt.show()

# ----------------------------
# 5. Line Chart - Trend of Flares Over Time
# ----------------------------
df_sorted = df_clean.sort_values('beginTime')
plt.figure(figsize=(12,6))
plt.plot(df_sorted['beginTime'], df_sorted['duration_minutes'], marker='', linestyle='-', alpha=0.6)
plt.title("Line Chart: Flare Duration Over Time (Cleaned)")
plt.xlabel("Date")
plt.ylabel("Duration (minutes)")
plt.show()

# ----------------------------
# 6. Scatter Plot - Duration vs Time
# ----------------------------
plt.figure(figsize=(10,6))
sns.scatterplot(
    x='beginTime', 
    y='duration_minutes', 
    hue='classType', 
    data=df_clean, 
    alpha=0.7, 
    palette="tab10", 
    s=50
)
plt.title("Scatter Plot of Flare Duration Over Time")
plt.xlabel("Start Time")
plt.ylabel("Duration (minutes)")
plt.legend(title="Flare Class", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Optional: Scatter plot with Flux if column exists
if 'flux' in df.columns:
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='duration_minutes', y='flux', hue='classType', data=df_clean, alpha=0.7, palette="coolwarm")
    plt.title("Scatter Plot: Duration vs Flux")
    plt.xlabel("Duration (minutes)")
    plt.ylabel("Flux")
    plt.show()

# ----------------------------
# 7. Stats Summary
# ----------------------------
print("\n--- Cleaned Flare Duration Statistics ---")
print(df_clean['duration_minutes'].describe())

print("\n--- Flare Duration Statistics (minutes) ---")
print("Mean:", df['duration_minutes'].mean())
print("Median:", df['duration_minutes'].median())
print("Std Dev:", df['duration_minutes'].std())
print("Min:", df['duration_minutes'].min())
print("Max:", df['duration_minutes'].max())
print("Range:", df['duration_minutes'].max() - df['duration_minutes'].min())

print("\n--- Flare Class Stats ---")
print("Mode:", df['classType'].mode()[0])
print(df['classType'].value_counts())

# ----------------------------
# 8. Scatter Plot of Flare Locations
# ----------------------------

# Regex for parsing heliographic coordinates like S25W03
location_pattern = re.compile(r'([NS])(\d+)([EW])(\d+)')

def parse_location(location):
    if not isinstance(location, str):
        return None, None
    match = location_pattern.match(location)
    if not match:
        return None, None
    ns_dir, ns_deg, ew_dir, ew_deg = match.groups()
    lat = int(ns_deg) if ns_dir == 'N' else -int(ns_deg)
    lon = int(ew_deg) if ew_dir == 'E' else -int(ew_deg)  # E positive, W negative
    return lat, lon

# Apply to DataFrame
df_clean[['latitude', 'longitude']] = df_clean['sourceLocation'].apply(lambda x: pd.Series(parse_location(x)))

# Drop rows where parsing failed
df_clean = df_clean.dropna(subset=['latitude', 'longitude'])

# Scatter plot of flare locations, colored by class
plt.figure(figsize=(10,6))
sns.scatterplot(
    x='longitude', y='latitude', 
    hue='classType', data=df_clean,
    alpha=0.6, palette="tab20", s=40
)
plt.title("Geographical Distribution of Solar Flare Locations")
plt.xlabel("Solar Longitude (°) (East + / West -)")
plt.ylabel("Solar Latitude (°) (North + / South -)")
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.axvline(0, color='black', linestyle='--', linewidth=0.7)
plt.legend(title="Flare Class", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()
