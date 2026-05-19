"""Test accuracy without Economic Impact to make climate matter."""
import pandas as pd
import numpy as np
import os
from pipeline import load_and_preprocess
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

csv = os.path.abspath(os.path.join('.', '..', 'climate_change_impact_on_agriculture_2024 (1).csv'))
df, log = load_and_preprocess(csv)

yield_col = 'Crop_Yield_MT_per_HA'
q_low = df[yield_col].quantile(0.30)
q_high = df[yield_col].quantile(0.70)

df_filtered = df[(df[yield_col] <= q_low) | (df[yield_col] >= q_high)].copy()
df_filtered.reset_index(drop=True, inplace=True)
y = (df_filtered[yield_col] >= q_high).map({True: 'Tinggi', False: 'Rendah'})

climate_features = [
    'Year', 'Average_Temperature_C', 'Total_Precipitation_mm', 
    'CO2_Emissions_MT', 'Temp_Category'
]

X = df_filtered[climate_features]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gb = GradientBoostingClassifier(n_estimators=300, max_depth=7, learning_rate=0.1, subsample=0.8, random_state=42)
acc = cross_val_score(gb, X, y, cv=cv, scoring='accuracy').mean()
print(f"Accuracy with Climate features ONLY: {acc:.4f}")
