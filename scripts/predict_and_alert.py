import joblib
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
model_dir = project_root / "models"
model_path = model_dir / "model_rf.joblib"

# Load the trained model
model = joblib.load(model_path)

def main() -> None:
    """
    Main function to get user input, predict solar flare risk, and print results.
    """
    # Prompt user for flux input until a valid float is entered
    while True:
        try:
            new_flux_value = float(input("Enter current X-ray flux value (e.g., 5e-07): "))
            break
        except ValueError:
            print("Invalid input. Please enter a numeric flux value.")

    # Prepare input for prediction as DataFrame
    X_new = pd.DataFrame([[new_flux_value]], columns=['flux'])

    # Predict flare class number
    predicted_class_num = model.predict(X_new)[0]

    # Get prediction probabilities if available
    try:
        probs = model.predict_proba(X_new)[0]
    except AttributeError:
        probs = None

    # Mapping from model output labels to flare classes (including 'No Flare')
    class_map_rev = {
        0: 'No Flare',
        1: 'B',
        2: 'C',
        3: 'M',
        4: 'X'
    }

    risk_class = class_map_rev.get(predicted_class_num, 'Unknown')

    # Define alert levels based on flare class severity
    alert_levels = {
        'No Flare': 'No Risk',
        'B': 'Low Risk',
        'C': 'Moderate Risk',
        'M': 'High Risk',
        'X': 'Severe Risk - Immediate Attention Required'
    }

    alert = alert_levels.get(risk_class, "No alert")

    # Print results
    print(f"\nInput flux value: {new_flux_value}")
    print(f"Predicted Flare Class: {risk_class}")
    print(f"Risk Alert: {alert}")
    if probs is not None:
        # Format and print probabilities nicely with flare class labels
        prob_str = ", ".join(
            f"{class_map_rev[i]}: {prob:.2%}" for i, prob in enumerate(probs)
        )
        print(f"Prediction probabilities: {prob_str}")

if __name__ == "__main__":
    main()
