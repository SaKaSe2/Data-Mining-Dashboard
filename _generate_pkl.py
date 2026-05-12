import os
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import load_and_preprocess, prepare_mining_data, run_chi2_selection, run_experiment, DATA_DIR

CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'climate_change_impact_on_agriculture_2024 (1).csv'))

print("Memuat dan preprocessing data...")
df, log = load_and_preprocess(CSV)
df, nf, y, lc, dist = prepare_mining_data(df)

print("Seleksi Fitur (Chi-Square)...")
chi2_f, _, _ = run_chi2_selection(df, nf, y)

print("Melatih Model Random Forest terbaik...")
tuned_params = {
    'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100,
    'random_state': 42, 'class_weight': 'balanced', 'n_jobs': -1
}
_, _, _, m4 = run_experiment('Eks-4: RF + Tuned', 4, df[chi2_f].copy(), y, lc, rf_params=tuned_params)

print("Menyimpan ke .pkl...")
joblib.dump(m4, os.path.join(DATA_DIR, 'best_model.pkl'))
joblib.dump(log['transformers'], os.path.join(DATA_DIR, 'transformers.pkl'))
joblib.dump(chi2_f, os.path.join(DATA_DIR, 'chi2_features.pkl'))
print("Selesai!")
