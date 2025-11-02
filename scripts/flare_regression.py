# flare_regression.py
import re
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def parse_source_location(loc):
    if pd.isna(loc):
        return [np.nan, np.nan]
    loc = str(loc)
    m = re.match(r'([NS])(\d+)([EW])(\d+)', loc)
    if not m:
        return [np.nan, np.nan]
    lat_dir, lat_deg, lon_dir, lon_deg = m.groups()
    lat = float(lat_deg) * (1 if lat_dir == 'N' else -1)
    lon = float(lon_deg) * (1 if lon_dir == 'E' else -1)
    return [lat, lon]

def load_and_feature_engineer():
    df = pd.read_csv('./nasa_flare_events_2010_2025.csv',
                     parse_dates=['beginTime','peakTime','submissionTime'])
    df['endTime'] = pd.to_datetime(df['endTime'], errors='coerce')
    df = df.dropna(subset=['classNumeric'])
    df = df.dropna(subset=['beginTime','peakTime','endTime'])
    df['classNumeric'] = pd.to_numeric(df['classNumeric'], errors='coerce')
    df['rise_minutes'] = (df['peakTime'] - df['beginTime']).dt.total_seconds() / 60.0
    df['decay_minutes'] = (df['endTime'] - df['peakTime']).dt.total_seconds() / 60.0
    df['total_duration_minutes'] = (df['endTime'] - df['beginTime']).dt.total_seconds() / 60.0
    df['begin_hour'] = df['beginTime'].dt.hour
    df['begin_dayofyear'] = df['beginTime'].dt.dayofyear
    df['begin_year'] = df['beginTime'].dt.year
    df['sourceLocation'] = df['sourceLocation'].astype(str)
    df['src_lat'] = df['sourceLocation'].apply(lambda x: parse_source_location(x)[0])
    df['src_lon'] = df['sourceLocation'].apply(lambda x: parse_source_location(x)[1])
    df['activeRegionNum'] = pd.to_numeric(df['activeRegionNum'], errors='coerce')
    df['classPrefix'] = df['classType'].str.extract(r'([A-Z])', expand=False)
    return df

def build_feature_matrix(df):
    numeric_features = ['rise_minutes','decay_minutes','total_duration_minutes','begin_hour','begin_dayofyear','src_lat','src_lon','activeRegionNum']
    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                    ('scaler', StandardScaler())])
    categorical_features = ['catalog','classPrefix']
    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ]
    )
    X = df[numeric_features + categorical_features]
    y = df['classNumeric']
    return X, y, preprocessor

def main():
    df = load_and_feature_engineer()
    X, y, preprocessor = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline_lr = Pipeline([('pre', preprocessor), ('lr', LinearRegression())])
    pipeline_lr.fit(X_train, y_train)
    y_pred = pipeline_lr.predict(X_test)
    print(f"Linear Regression RMSE: {mean_squared_error(y_test, y_pred, squared=False):.6e}")
    print(f"Linear Regression R²: {r2_score(y_test, y_pred):.4f}")

    pipeline_ridge = Pipeline([('pre', preprocessor), ('ridge', Ridge(alpha=1.0))])
    pipeline_ridge.fit(X_train, y_train)
    y_pred_ridge = pipeline_ridge.predict(X_test)
    print(f"Ridge RMSE: {mean_squared_error(y_test, y_pred_ridge, squared=False):.6e}")
    print(f"Ridge R²: {r2_score(y_test, y_pred_ridge):.4f}")

    joblib.dump(pipeline_lr, './flare_linear_regression.joblib')
    joblib.dump(pipeline_ridge, './flare_ridge_regression.joblib')
    print("Models saved.")

if __name__ == '__main__':
    main()
