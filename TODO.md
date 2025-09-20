# Solar Flare Prediction Enhancement Plan

## Overview
Enhance the Streamlit app by integrating Prophet for accurate flux prediction on specific dates, improving upon the current historical average method.

## Steps
- [x] Update scripts/streamlit_app.py to import Prophet and load historical data for training
- [x] Add Prophet model training function in the app
- [x] Modify the prediction logic to use Prophet for flux prediction instead of historical average
- [x] Update the UI to display predicted flux and flare class with danger level
- [ ] Test the updated Streamlit app
- [ ] Run the app to verify functionality

## Dependent Files
- scripts/streamlit_app.py (main file to edit)
- data/historical_goes_2010_2015_parsed.csv (data source)
- models/model_rf_improved.joblib (RF model for flare class)

## Followup Steps
- Install any missing dependencies if needed
- Verify Prophet is installed in the environment
- Test with sample dates
