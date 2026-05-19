import os
import sys
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fastapi import FastAPI, Request, Form, Body
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import load_and_preprocess, preprocess_new_data, DATA_DIR

app = FastAPI(title="Data Mining Dashboard - Tabler")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- GLOBAL VARIABLES & STARTUP ---
MODEL = None
FEATURES = None
CAT_FEATURES = None
MEDIANS = {}
CSV_PATH = os.path.join(BASE_DIR, 'data', 'train', 'train_data.csv')
DF_VIZ = None
LAST_STATE = None

@app.on_event("startup")
def load_assets():
    global MODEL, FEATURES, CAT_FEATURES, MEDIANS, DF_VIZ
    try:
        MODEL = CatBoostClassifier()
        EKSPORT_DIR = os.path.join(DATA_DIR, 'eksport')
        MODEL.load_model(os.path.join(EKSPORT_DIR, 'catboost_model.cbm'))
        FEATURES = joblib.load(os.path.join(EKSPORT_DIR, 'catboost_features.pkl'))
        CAT_FEATURES = joblib.load(os.path.join(EKSPORT_DIR, 'catboost_cat_features.pkl'))
        
        # Load raw data untuk medians dan context viz
        df_raw = pd.read_csv(CSV_PATH, delimiter=';')
        for col in df_raw.select_dtypes(include=np.number).columns:
            MEDIANS[col] = df_raw[col].median()
            
        DF_VIZ = df_raw.copy()
        for col in df_raw.select_dtypes(include=np.number).columns:
            DF_VIZ[col].fillna(DF_VIZ[col].median(), inplace=True)
            
        print("[OK] CatBoost Model & Features Loaded Successfully!")
    except Exception as e:
        print(f"[WARNING] Error loading models: {e}")
        print("Pastikan Anda sudah menjalankan 'python _generate_catboost.py'")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Halaman Utama (Simulasi Prediksi)."""
    context = {
        "request": request, 
        "title": "Dashboard Prediksi",
        "has_result": False
    }
    if LAST_STATE:
        context.update(LAST_STATE)
        
    return templates.TemplateResponse(request=request, name="index.html", context=context)


@app.post("/predict", response_class=HTMLResponse)
async def predict_api(
    request: Request,
    temp: float = Form(...),
    precip: float = Form(...),
    co2: float = Form(...),
    economic: float = Form(...)
):
    """Logika Prediksi dan Visualisasi."""
    global MODEL, FEATURES, CAT_FEATURES, MEDIANS, DF_VIZ, LAST_STATE
    
    # 1. Siapkan input dasar
    input_data = {
        'Average_Temperature_C': temp,
        'Total_Precipitation_mm': precip,
        'CO2_Emissions_MT': co2,
        'Economic_Impact_Million_USD': economic,
        'Country': 'Indonesia', # Default asumsi untuk skenario dashboard
        'Region': 'Jawa Timur',
        'Crop_Type': 'Rice',
        'Extreme_Weather_Events': MEDIANS.get('Extreme_Weather_Events', 1),
        'Irrigation_Access_%': MEDIANS.get('Irrigation_Access_%', 50),
        'Pesticide_Use_KG_per_HA': MEDIANS.get('Pesticide_Use_KG_per_HA', 10),
        'Fertilizer_Use_KG_per_HA': MEDIANS.get('Fertilizer_Use_KG_per_HA', 30),
        'Soil_Health_Index': MEDIANS.get('Soil_Health_Index', 60),
    }
    
    # 2. Isi fitur numerik lain yang kosong dengan median
    for f in FEATURES:
        if f not in input_data and f not in CAT_FEATURES:
            input_data[f] = MEDIANS.get(f, 0.0)
            
    df_input_raw = pd.DataFrame([input_data])
    
    # 3. Preprocessing (Feature Crosses, Hemisphere, dll)
    df_processed = preprocess_new_data(df_input_raw, FEATURES, CAT_FEATURES)
    
    # 4. Prediksi
    pred_class = MODEL.predict(df_processed)
    if len(pred_class.shape) > 1:
        pred_class = pred_class.flatten()
    pred_class = pred_class[0]
    
    proba = MODEL.predict_proba(df_processed)[0]
    class_idx = list(MODEL.classes_).index(pred_class)
    conf = proba[class_idx] * 100
    
    hasil_teks = pred_class
    
    # 5. Siapkan data untuk chart di template (ApexCharts)
    # Data scatter: ambil sample historis
    df_sample = DF_VIZ.sample(n=min(100, len(DF_VIZ)), random_state=42)
    scatter_hist = [
        {"x": round(float(row['Average_Temperature_C']), 2), "y": round(float(row['Total_Precipitation_mm']), 2)}
        for _, row in df_sample.iterrows()
    ]
    
    # Simpan state untuk halaman Eksperimen
    LAST_STATE = {
        "has_result": True,
        "prediction_result": hasil_teks,
        "confidence": f"{conf:.1f}%",
        "conf_value": round(conf, 1),
        "scatter_data": scatter_hist,
        "temp": temp, "precip": precip, "co2": co2, "economic": economic,
        "medians": MEDIANS
    }
    
    context = {
        "request": request, 
        "title": "Hasil Prediksi",
    }
    context.update(LAST_STATE)
    
    # Return ke template
    return templates.TemplateResponse(request=request, name="index.html", context=context)


@app.get("/eksperimen", response_class=HTMLResponse)
async def read_eksperimen(request: Request):
    """Halaman Eksperimen (Training Model)."""
    context = {
        "request": request, 
        "title": "Eksperimen Data Mining"
    }
    if LAST_STATE:
        context.update(LAST_STATE)
        
    return templates.TemplateResponse(request=request, name="eksperimen.html", context=context)

@app.post("/api/predict_single")
async def api_predict_single(data: dict = Body(...)):
    global MODEL, FEATURES, CAT_FEATURES, MEDIANS
    # Jika tidak ada input yang diberikan, buat input default
    df_input = pd.DataFrame([data])
    for f in FEATURES:
        if f not in df_input.columns and f not in CAT_FEATURES:
            df_input[f] = MEDIANS.get(f, 0.0)
            
    df_processed = preprocess_new_data(df_input, FEATURES, CAT_FEATURES)
    
    pred_class = MODEL.predict(df_processed)
    if len(pred_class.shape) > 1: pred_class = pred_class.flatten()
    pred_class = pred_class[0]
    
    proba = MODEL.predict_proba(df_processed)[0]
    class_idx = list(MODEL.classes_).index(pred_class)
    confidence = proba[class_idx] * 100
    
    return {"prediction": pred_class, "confidence": round(confidence, 2)}

@app.get("/api/predict_batch")
async def api_predict_batch():
    global MODEL, FEATURES, CAT_FEATURES, DF_VIZ
    # Simulasi seperti di ipynb: ambil 5 data secara acak dari DF_VIZ
    df_sample = DF_VIZ.sample(n=5)
    df_input = df_sample.drop(columns=['Crop_Yield_MT_per_HA', 'Yield_Class_Binary'], errors='ignore')
    
    df_processed = preprocess_new_data(df_input, FEATURES, CAT_FEATURES)
    
    preds = MODEL.predict(df_processed)
    probas = MODEL.predict_proba(df_processed)
    classes = list(MODEL.classes_)
    
    results = []
    for i in range(len(df_sample)):
        p_class = preds[i]
        if isinstance(p_class, (list, np.ndarray)): p_class = p_class[0]
        c_idx = classes.index(p_class)
        conf = probas[i][c_idx] * 100
        
        row_dict = df_sample.iloc[i].to_dict()
        results.append({
            "Country": row_dict.get("Country", "-"),
            "Crop_Type": row_dict.get("Crop_Type", "-"),
            "Average_Temperature_C": round(float(row_dict.get("Average_Temperature_C", 0)), 2),
            "Total_Precipitation_mm": round(float(row_dict.get("Total_Precipitation_mm", 0)), 2),
            "Fertilizer_Use_KG_per_HA": round(float(row_dict.get("Fertilizer_Use_KG_per_HA", 0)), 2),
            "Prediction": p_class,
            "Confidence": round(conf, 2)
        })
    return {"batch_results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
