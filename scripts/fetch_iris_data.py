import shutil
import os

# Source: where your dataset is currently located
source_path = r"C:\Users\Haneesh.B\OneDrive\Documents\IRISMIL_dataset_10000_bags.npz"

# Destination: your project data folder
destination_folder = r"C:\Users\Haneesh.B\Documents\SOLAR_FLARE_NEW\data"
os.makedirs(destination_folder, exist_ok=True)

# Final filename in project
destination_path = os.path.join(destination_folder, "iris_solar_flare_dataset.npz")

# Move the file (use copy if you want to keep the original)
shutil.move(source_path, destination_path)

print("File moved successfully to:", destination_path)
