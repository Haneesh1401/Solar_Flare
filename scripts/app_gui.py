import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "models" / "model_rf.joblib"
model = joblib.load(model_path)

class_map_rev = {1: 'B', 2: 'C', 3: 'M', 4: 'X'}
alert_levels = {
    'B': 'Low Risk',
    'C': 'Moderate Risk',
    'M': 'High Risk',
    'X': 'Severe Risk - Immediate Attention Required'
}

def predict():
    try:
        flux_val = float(entry.get())
        X_new = pd.DataFrame([[flux_val]], columns=['flux'])
        pred = model.predict(X_new)[0]
        risk = class_map_rev.get(pred, 'Unknown')
        alert = alert_levels.get(risk, "No alert")
        messagebox.showinfo("Prediction Result", f"Flare Class: {risk}\nAlert: {alert}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid number.")

root = tk.Tk()
root.title("Solar Flare Risk Predictor")

tk.Label(root, text="Enter X-ray Flux Value:").pack(pady=10)
entry = tk.Entry(root)
entry.pack(pady=5)

btn = tk.Button(root, text="Predict Risk", command=predict)
btn.pack(pady=20)

root.mainloop()
