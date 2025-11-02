import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set seaborn style for better visuals
sns.set_style('whitegrid')

# Set script directory
script_dir = Path(__file__).resolve().parent

# Load dataset
csv_file = script_dir / "historical_goes_2010_2025_cleaned.csv"
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

# ------------------------------
# 5. Heat map for flare counts by year and month
# ------------------------------
if "year" in df.columns and "month" in df.columns:
    pivot_table = df.groupby(['year', 'month']).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Number of Flares'})
    plt.title("Heat Map of Solar Flare Counts by Year and Month")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.tight_layout()
    plt.savefig(script_dir / "flare_heatmap_year_month.png", dpi=300)
    plt.show()

# ------------------------------
# 6. Correlation heat map for numerical columns
# ------------------------------
numerical_cols = ['classNumeric', 'activeRegionNum']
available_num_cols = [col for col in numerical_cols if col in df.columns]
if available_num_cols:
    corr_matrix = df[available_num_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
    plt.title("Correlation Heat Map of Numerical Features")
    plt.tight_layout()
    plt.savefig(script_dir / "correlation_heatmap.png", dpi=300)
    plt.show()

print("\n✅ Graphs saved in the same folder as this script.")
