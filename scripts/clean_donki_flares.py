import pandas as pd
import ast

# Load your raw CSV
df = pd.read_csv("scripts/nasa_flare_events_2010_2025.csv")

# Flatten 'instruments' column to extract displayName
def extract_instrument(instr):
    try:
        instr_list = ast.literal_eval(instr)
        return ", ".join([i['displayName'] for i in instr_list])
    except:
        return None

df['instruments'] = df['instruments'].apply(extract_instrument)

# Flatten 'linkedEvents' column to extract activityID
def extract_linked_events(events):
    try:
        events_list = ast.literal_eval(events)
        return ", ".join([e['activityID'] for e in events_list])
    except:
        return None

df['linkedEvents'] = df['linkedEvents'].apply(extract_linked_events)

# Flatten 'sentNotifications' column to extract messageID
def extract_notifications(notes):
    try:
        notes_list = ast.literal_eval(notes)
        return ", ".join([n['messageID'] for n in notes_list])
    except:
        return None

df['sentNotifications'] = df['sentNotifications'].apply(extract_notifications)

# Convert time columns to datetime
time_cols = ['beginTime', 'peakTime', 'endTime', 'submissionTime']
for col in time_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Optional: convert flare class to numeric scale (W/m²)
def flare_class_to_numeric(flt_class):
    if pd.isna(flt_class):
        return None
    factor = {'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}
    letter = flt_class[0]
    number = float(flt_class[1:])
    return factor.get(letter, None) * number

df['classNumeric'] = df['classType'].apply(flare_class_to_numeric)

# Save cleaned CSV
df.to_csv("historical_goes_2010_2025_cleaned.csv", index=False)

print("✅ Cleaned CSV saved as 'historical_goes_2010_2025_cleaned.csv'")
