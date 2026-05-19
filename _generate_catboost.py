import pandas as pd
import numpy as np
import os
import joblib
from catboost import CatBoostClassifier

# Path ke file CSV baru (harus di root project atau disesuaikan)
CSV_PATH = 'data/train/train_data.csv'
MODEL_PATH = 'data/eksport/catboost_model.cbm'
FEATURES_PATH = 'data/eksport/catboost_features.pkl'

def clean_numeric_dots(x):
    if isinstance(x, str):
        if x.count('.') > 1:
            parts = x.replace('.', '')
            try:
                val = float(parts)
                while val > 100:  
                    val = val / 10
                return val
            except:
                return np.nan
        else:
            try:
                return float(x)
            except:
                return np.nan
    return x

def preprocess_train_data(df):
    df_new = df.copy()
    
    # 1. Bersihkan numerik & fill missing
    if 'Crop_Yield_MT_per_HA' in df_new.columns:
        df_new['Crop_Yield_MT_per_HA'] = pd.to_numeric(df_new['Crop_Yield_MT_per_HA'].apply(clean_numeric_dots))
    if 'Extreme_Weather_Events' in df_new.columns:
        df_new['Extreme_Weather_Events'] = pd.to_numeric(df_new['Extreme_Weather_Events'], errors='coerce')

    for col in df_new.select_dtypes(include=np.number).columns:
        df_new[col].fillna(df_new[col].median(), inplace=True)

    # 2. Hemisphere
    hemisphere_map = {
        'India': 'Northern', 'China': 'Northern', 'France': 'Northern', 'USA': 'Northern', 
        'Canada': 'Northern', 'Russia': 'Northern', 'Nigeria': 'Northern', 
        'Australia': 'Southern', 'Argentina': 'Southern', 'Brazil': 'Southern'
    }
    if 'Country' in df_new.columns:
        df_new['Hemisphere'] = df_new['Country'].map(hemisphere_map).fillna('Unknown')
    
    if 'Economic_Impact_Million_USD' in df_new.columns:
        df_new['Economic_Impact_Log'] = np.log1p(df_new['Economic_Impact_Million_USD'])

    # 3. Feature Crosses
    if 'Average_Temperature_C' in df_new.columns and 'Total_Precipitation_mm' in df_new.columns:
        df_new['Temperature_Precipitation_Ratio'] = df_new['Average_Temperature_C'] / (df_new['Total_Precipitation_mm'] + 1e-5)
        df_new['Temp_x_Precipitation'] = df_new['Average_Temperature_C'] * df_new['Total_Precipitation_mm']
        
    if 'Fertilizer_Use_KG_per_HA' in df_new.columns and 'Total_Precipitation_mm' in df_new.columns:
        df_new['Fertilizer_per_Precipitation'] = df_new['Fertilizer_Use_KG_per_HA'] / (df_new['Total_Precipitation_mm'] + 1e-5)
        
    if 'Pesticide_Use_KG_per_HA' in df_new.columns and 'Total_Precipitation_mm' in df_new.columns:
        df_new['Pesticide_per_Precipitation'] = df_new['Pesticide_Use_KG_per_HA'] / (df_new['Total_Precipitation_mm'] + 1e-5)

    if 'Soil_Health_Index' in df_new.columns and 'Fertilizer_Use_KG_per_HA' in df_new.columns:
        df_new['Soil_x_Fertilizer'] = df_new['Soil_Health_Index'] * df_new['Fertilizer_Use_KG_per_HA']
        
    if 'Total_Precipitation_mm' in df_new.columns:
        df_new['Log_Precipitation'] = np.log1p(df_new['Total_Precipitation_mm'])
        if 'Economic_Impact_Log' in df_new.columns:
            df_new['Economic_x_Precip'] = df_new['Economic_Impact_Log'] * df_new['Total_Precipitation_mm']
            
    if 'Average_Temperature_C' in df_new.columns and 'Soil_Health_Index' in df_new.columns:
        df_new['Temp_x_Soil'] = df_new['Average_Temperature_C'] * df_new['Soil_Health_Index']

    if all(k in df_new.columns for k in ['Country', 'Region', 'Crop_Type']):
        df_new['Country_Region_Crop'] = df_new['Country'].astype(str) + '_' + df_new['Region'].astype(str) + '_' + df_new['Crop_Type'].astype(str)
        
    if all(k in df_new.columns for k in ['Extreme_Weather_Events', 'Total_Precipitation_mm', 'Irrigation_Access_%']):
        df_new['Climate_Stress_Index'] = (df_new['Extreme_Weather_Events'] * df_new['Total_Precipitation_mm']) / (df_new['Irrigation_Access_%'] + 1e-5)

    # 4. Target Biner
    mean_yield = df_new['Crop_Yield_MT_per_HA'].median()
    df_new['Yield_Class_Binary'] = np.where(df_new['Crop_Yield_MT_per_HA'] > mean_yield, 'Di Atas Rata-rata', 'Di Bawah Rata-rata')
    
    return df_new

def main():
    print("Membaca dataset...")
    df_raw = pd.read_csv(CSV_PATH, delimiter=';')
    
    print("Pra-pemrosesan Data (Feature Engineering Tahap 6)...")
    df = preprocess_train_data(df_raw)
    
    # 5. Siapkan Fitur dan Target
    drop_cols = ['Crop_Yield_MT_per_HA', 'Yield_Class_Binary']
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['Yield_Class_Binary']
    
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_features:
        X[col] = X[col].astype(str)
        
    # Fitur numerik sisanya (pastikan bertipe float)
    num_features = [col for col in X.columns if col not in cat_features]
    for col in num_features:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
    print(f"Melatih CatBoost dengan {len(X.columns)} fitur ({len(cat_features)} kategorikal)...")
    
    # Inisialisasi model
    cat_model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=5,
        bagging_temperature=0.2,
        eval_metric='Accuracy',
        random_seed=42,
        verbose=100,
        cat_features=cat_features
    )
    
    cat_model.fit(X, y)
    
    # Simpan model
    os.makedirs('data/eksport', exist_ok=True)
    cat_model.save_model(MODEL_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)
    joblib.dump(cat_features, 'data/eksport/catboost_cat_features.pkl')
    
    print(f"[SUCCESS] Model disimpan ke '{MODEL_PATH}'")
    print(f"[SUCCESS] Daftar fitur disimpan ke '{FEATURES_PATH}'")

if __name__ == '__main__':
    main()
