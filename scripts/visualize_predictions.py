import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set script directory
script_dir = Path(__file__).resolve().parent

# Load dataset
csv_file = script_dir / "nasa_flare_events_2010_2025.csv"
df = pd.read_csv(csv_file)

# Show dataset structure
print("\n--- Available Columns ---")
print(df.columns)

print("\n--- First 5 Rows ---")
print(df.head())



if "classType" in df.columns and "beginTime" in df.columns:
    df["beginTime"] = pd.to_datetime(df["beginTime"], errors="coerce")
    df = df.dropna(subset=["beginTime", "classType"])
    df["year"] = df["beginTime"].dt.year
    class_counts = df.groupby(["year", "classType"]).size().unstack(fill_value=0)

    class_counts.plot(kind="bar", stacked=True, figsize=(14, 6))
    plt.title("Solar Flare Class Distribution by Year", fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.legend(title="Flare Class")
    plt.tight_layout()

    output_file = script_dir / "flare_class_distribution.png"
    plt.savefig(output_file, dpi=300)
    print(f"\nPlot saved as {output_file}")

    plt.show()
else:
    print("\n⚠️ Required columns for plotting not found in dataset.")

# Convert time column if available
if "beginTime" in df.columns:
    df["beginTime"] = pd.to_datetime(df["beginTime"], errors="coerce")
    df["year"] = df["beginTime"].dt.year
    df["month"] = df["beginTime"].dt.month

# ------------------------------
# 1. Flare counts per year
# ------------------------------
if "year" in df.columns:
    yearly_counts = df["year"].value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    yearly_counts.plot(kind="bar")
    plt.title("Number of Solar Flares per Year")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(script_dir / "flares_per_year.png", dpi=300)
    plt.show()

# ------------------------------
# 2. Distribution of flare classes
# ------------------------------
if "classType" in df.columns:
    plt.figure(figsize=(8, 5))
    df["classType"].value_counts().plot(kind="bar", color="orange")
    plt.title("Distribution of Flare Classes")
    plt.xlabel("Flare Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(script_dir / "flare_class_distribution.png", dpi=300)
    plt.show()

# ------------------------------
# 3. Monthly trend (seasonality)
# ------------------------------
if "month" in df.columns:
    monthly_counts = df["month"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    monthly_counts.plot(kind="line", marker="o")
    plt.title("Monthly Solar Flare Frequency")
    plt.xlabel("Month")
    plt.ylabel("Number of Flares")
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(script_dir / "monthly_trend.png", dpi=300)
    plt.show()

# ------------------------------
# 4. Scatter plot of flares over time
# ------------------------------
if "beginTime" in df.columns and "classType" in df.columns:
    plt.figure(figsize=(12, 6))
    plt.scatter(df["beginTime"], df["classType"], alpha=0.6, s=20)
    plt.title("Solar Flare Classes Over Time")
    plt.xlabel("Date")
    plt.ylabel("Flare Class")
    plt.tight_layout()
    plt.savefig(script_dir / "flare_scatter_time.png", dpi=300)
    plt.show()

print("\n✅ Graphs saved in the same folder as this script.")



# Histogram of Peak Intensity
plt.figure(figsize=(8, 5))
plt.hist(df['peak_intensity'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel("Peak Intensity")
plt.ylabel("Frequency")
plt.title("Distribution of Peak Intensity")
plt.tight_layout()
plt.show()

# Top 20 Flare Classes
plt.figure(figsize=(10, 6))
df['flare_class'].value_counts().head(20).plot(kind='bar', color='orange')
plt.xlabel("Flare Class")
plt.ylabel("Count")
plt.title("Top 20 Flare Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()