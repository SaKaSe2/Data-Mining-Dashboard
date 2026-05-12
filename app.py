"""
Dashboard Data Mining - Climate Change Impact on Agriculture.
Streamlit app dengan 2 menu utama: Eksperimen & Hasil, dan Simulasi Prediksi.
Mendukung export/import model .pkl dan loading indicator yang lebih baik.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from pipeline import (
    load_and_preprocess, prepare_mining_data,
    run_chi2_selection, run_rfe_selection,
    run_experiment, run_gridsearch,
    create_comparison_chart, plot_target_distribution,
    plot_class_distribution, DATA_DIR
)

# --- Konfigurasi halaman ---
st.set_page_config(
    page_title="Dashboard Data Mining - Agriculture",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling (Tabler Theme) ---
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #f4f6fa;
        color: #232e3c;
    }
    
    /* Typography & Headers */
    .main-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #182433;
        margin-bottom: 1.5rem;
        letter-spacing: -0.01em;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(98, 105, 118, 0.16);
    }
    
    /* Tabler Card Style for Metrics & Layouts */
    .metric-card, div[data-testid="stForm"] {
        background: #ffffff !important;
        border: 1px solid rgba(98, 105, 118, 0.16) !important;
        border-radius: 4px !important;
        padding: 1.25rem !important;
        box-shadow: rgba(0, 0, 0, 0.04) 0px 2px 4px 0px !important;
        color: #232e3c !important;
    }
    .metric-card h3 { 
        margin: 0; 
        font-size: 0.75rem; 
        font-weight: 600; 
        color: #626976; 
        text-transform: uppercase; 
        letter-spacing: 0.04em; 
    }
    .metric-card h1 { 
        margin: 0.5rem 0 0 0; 
        font-size: 1.5rem; 
        font-weight: 600; 
        color: #182433; 
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #206bc4 !important;
        color: white !important;
        border-radius: 4px !important;
        border: 1px solid transparent !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
        box-shadow: rgba(0, 0, 0, 0.04) 0px 2px 4px 0px !important;
    }
    .stButton>button:hover {
        background-color: #1a569d !important;
        border-color: #1a569d !important;
    }
    
    /* Inputs */
    div[data-baseweb="input"] > div {
        border-radius: 4px !important;
        border: 1px solid rgba(98, 105, 118, 0.16) !important;
        background-color: #ffffff !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(98, 105, 118, 0.16) !important;
        background-color: #ffffff !important;
    }
    
    /* Expander / Accordion */
    .streamlit-expanderHeader {
        border-radius: 4px !important;
        background-color: #ffffff !important;
        border: 1px solid rgba(98, 105, 118, 0.16) !important;
    }
    
    /* Alerts / Success Box (Tabler Alert Style) */
    .success-box {
        background: #e6f6e9;
        border-left: 3px solid #2fb344;
        padding: 1rem;
        border-radius: 4px;
        color: #232e3c;
        margin: 1rem 0;
        font-size: 0.875rem;
    }
    
    /* Prediction Result Box */
    .pred-box {
        background: #ffffff;
        border: 1px solid rgba(98, 105, 118, 0.16);
        box-shadow: rgba(0, 0, 0, 0.04) 0px 2px 4px 0px;
        padding: 2rem;
        border-radius: 4px;
        text-align: center;
        margin-top: 1rem;
    }
    .pred-box h4 {
        color: #626976;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .pred-text {
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# --- Path file ---
CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'climate_change_impact_on_agriculture_2024 (1).csv'))
MODEL_PATH = os.path.join(DATA_DIR, 'best_model.pkl')
TRANS_PATH = os.path.join(DATA_DIR, 'transformers.pkl')
CHI2_PATH = os.path.join(DATA_DIR, 'chi2_features.pkl')

# --- Session state management ---
def init_state():
    """Inisialisasi session state dan load PKL jika ada."""
    defaults = {
        'experiments_done': False,
        'df': None,
        'log': None,
        'chi2_features': None,
        'all_results': None,
        'comparison_path': None,
        'best_model': None,
        'from_pkl': False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Auto-load PKL jika ada dan belum dijalankan
    if not st.session_state.experiments_done and os.path.exists(MODEL_PATH) and os.path.exists(TRANS_PATH) and os.path.exists(CHI2_PATH):
        try:
            st.session_state.best_model = joblib.load(MODEL_PATH)
            st.session_state.log = {'transformers': joblib.load(TRANS_PATH)}
            st.session_state.chi2_features = joblib.load(CHI2_PATH)
            st.session_state.experiments_done = True
            st.session_state.from_pkl = True
        except Exception:
            pass

init_state()


# --- Sidebar ---
st.sidebar.markdown("## <i class='fa-solid fa-bars'></i> Menu Utama", unsafe_allow_html=True)
page = st.sidebar.radio("Pilih Halaman:", [
    "1. Eksperimen & Hasil",
    "2. Simulasi Prediksi (Uji Model)"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Climate Change Impact")
st.sidebar.markdown("**Algoritma:** Random Forest")

if st.session_state.experiments_done:
    if st.session_state.from_pkl:
        st.sidebar.markdown("<div style='color: #2fb344; font-weight: 600;'><i class='fa-solid fa-check-circle'></i> Model dimuat dari .pkl</div>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<div style='color: #2fb344; font-weight: 600;'><i class='fa-solid fa-check-circle'></i> Model selesai di-training!</div>", unsafe_allow_html=True)


# =====================================================================
# PAGE 1: EKSPERIMEN & HASIL
# =====================================================================
if page == "1. Eksperimen & Hasil":
    st.markdown('<div class="main-header">Jalankan Pipeline Data Mining</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Halaman ini akan menjalankan seluruh proses secara otomatis:
    1. **Preprocessing** (Cleaning, Transformasi, Normalisasi)
    2. **Feature Selection** (Chi-Square & RFE)
    3. **Training Model** (4 Eksperimen Random Forest termasuk Hyperparameter Tuning)
    """)

    if st.session_state.from_pkl:
        st.markdown("<div class='success-box'><i class='fa-solid fa-lightbulb'></i> <b>Model sudah dimuat dari file .pkl.</b> Kamu bisa langsung menuju halaman Simulasi Prediksi tanpa perlu men-training ulang. Namun jika ingin melatih ulang, silakan klik tombol di bawah.</div>", unsafe_allow_html=True)

    if st.button("Jalankan Ulang Semua Proses (Training & Save .pkl)", type="primary", use_container_width=True):
        
        with st.status("Sistem sedang memproses Pipeline Data Mining...", expanded=True) as status:
            try:
                # 1. Preprocess
                st.write("1/4: Memuat dan preprocessing data...")
                df, log = load_and_preprocess(CSV_PATH)
                df, nf, y, lc, dist = prepare_mining_data(df)
                st.session_state.df = df
                st.session_state.log = log
                plot_target_distribution(df)
                plot_class_distribution(y)
                
                # 2. Feature Selection
                st.write("2/4: Melakukan seleksi fitur (Chi-Square & RFE)...")
                chi2_f, chi2_df, chi2_path = run_chi2_selection(df, nf, y)
                rfe_f, rfe_df, rfe_path = run_rfe_selection(df, nf, y)
                st.session_state.chi2_features = chi2_f
                
                # 3. Eksperimen
                st.write("3/4: Menjalankan Baseline, Chi-Square, & RFE RandomForest (5-Fold CV)...")
                all_results = []
                r1, rep1, p1, m1 = run_experiment('Eks-1: RF Baseline', 1, df[nf].copy(), y, lc)
                all_results.append(r1)
                
                r2, rep2, p2, m2 = run_experiment('Eks-2: RF + Chi2', 2, df[chi2_f].copy(), y, lc)
                all_results.append(r2)
                
                r3, rep3, p3, m3 = run_experiment('Eks-3: RF + RFE', 3, df[rfe_f].copy(), y, lc)
                all_results.append(r3)
                
                # 4. GridSearchCV & Final Model
                st.write("4/4: Tuning Hyperparameter (GridSearchCV) - Memakan waktu beberapa saat...")
                best_params, best_score, top10 = run_gridsearch(df, chi2_f, y)
                tuned_params = best_params.copy()
                tuned_params.update({'random_state': 42, 'class_weight': 'balanced', 'n_jobs': None})
                
                st.write("Menjalankan model terbaik hasil tuning...")
                r4, rep4, p4, m4 = run_experiment('Eks-4: RF + Tuned', 4, df[chi2_f].copy(), y, lc, rf_params=tuned_params)
                all_results.append(r4)
                
                # Save to PKL
                st.write("Menyimpan model ke format .pkl...")
                joblib.dump(m4, MODEL_PATH)
                joblib.dump(log['transformers'], TRANS_PATH)
                joblib.dump(chi2_f, CHI2_PATH)

                # Simpan state & output
                results_df = pd.DataFrame(all_results).set_index('Eksperimen')
                comp_path = create_comparison_chart(results_df)
                results_df.to_csv(os.path.join(DATA_DIR, 'comparison_table.csv'))
                
                st.session_state.all_results = all_results
                st.session_state.comparison_path = comp_path
                st.session_state.best_model = m4  
                st.session_state.experiments_done = True
                st.session_state.from_pkl = False
                
                status.update(label="Training Selesai!", state="complete", expanded=False)
                st.markdown("<div class='success-box'><i class='fa-solid fa-check-circle'></i> Training model selesai! File `.pkl` berhasil diperbarui. Lihat tabel perbandingan di bawah.</div>", unsafe_allow_html=True)
            
            except Exception as e:
                status.update(label="Terjadi Kesalahan", state="error", expanded=True)
                st.error(f"Error detail saat memproses pipeline: {str(e)}")

    if st.session_state.experiments_done and not st.session_state.from_pkl and st.session_state.all_results is not None:
        st.markdown("---")
        st.markdown("### Tabel Perbandingan Performa")
        results = st.session_state.all_results
        results_df = pd.DataFrame(results).set_index('Eksperimen')
        st.dataframe(results_df.style.highlight_max(
            subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='#d5f4e6'
        ), use_container_width=True)

        st.markdown("### Visualisasi Perbandingan")
        if st.session_state.comparison_path:
            st.image(st.session_state.comparison_path, use_container_width=True)
            
        st.markdown("### Feature Importance (Model Terbaik)")
        fi_path = os.path.join(DATA_DIR, 'feature_importance_eks4.png')
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)
            
    elif st.session_state.from_pkl:
        st.markdown("---")
        st.markdown("### Feature Importance (Dari Model Terakhir)")
        fi_path = os.path.join(DATA_DIR, 'feature_importance_eks4.png')
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)
        else:
            st.info("Gambar komparasi tidak ditemukan, namun model sudah siap digunakan di halaman Simulasi.")


# =====================================================================
# PAGE 2: SIMULASI PREDIKSI (UJI MODEL)
# =====================================================================
elif page == "2. Simulasi Prediksi (Uji Model)":
    st.markdown('<div class="main-header">Simulasi Prediksi Hasil Panen</div>', unsafe_allow_html=True)
    
    if not st.session_state.experiments_done:
        st.markdown("<div style='color: #d63939; font-weight: 600;'><i class='fa-solid fa-exclamation-triangle'></i> Belum ada model yang tersedia! Silakan latih model terlebih dahulu di halaman 'Eksperimen & Hasil'.</div>", unsafe_allow_html=True)
        st.stop()

    if st.session_state.from_pkl:
        st.markdown("<div style='color: #206bc4; font-weight: 600; padding-bottom: 15px;'><i class='fa-solid fa-bolt'></i> Memuat model instan dari .pkl. Tidak perlu menunggu proses training ulang.</div>", unsafe_allow_html=True)

    # Ambil akurasi dari file CSV (jika ada)
    acc_text = "~ 62.13%"
    try:
        comp_csv = os.path.join(DATA_DIR, 'comparison_table.csv')
        if os.path.exists(comp_csv):
            comp_df = pd.read_csv(comp_csv, index_col=0)
            if 'Eks-4: RF + Tuned' in comp_df.index:
                acc = comp_df.loc['Eks-4: RF + Tuned', 'Accuracy']
                acc_text = f"{acc*100:.2f}%"
    except Exception:
        pass

    st.markdown(f"""
    Silakan masukkan data faktor lingkungan dan iklim di bawah ini. 
    Model **Random Forest (Tuned)** dengan **Akurasi Model: {acc_text}** akan memprediksi klasifikasi Hasil Panen berdasarkan pola yang telah dipelajari.
    """)

    # Form Input
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            year = st.number_input("Tahun (Year)", min_value=1990, max_value=2050, value=2024)
            temp = st.number_input("Rata-rata Suhu °C (Average_Temperature_C)", value=25.0)
            precip = st.number_input("Total Curah Hujan mm (Total_Precipitation_mm)", value=1500.0)
        with col2:
            co2 = st.number_input("Emisi CO2 MT (CO2_Emissions_MT)", value=15.0)
            economic = st.number_input("Dampak Ekonomi Juta USD (Economic_Impact_Million_USD)", value=800.0)
            
        submit = st.form_submit_button("Prediksi Hasil Panen", type="primary")

    if submit:
        # Load transformers & model
        transformers = st.session_state.log['transformers']
        kbins = transformers['kbins']
        scaler = transformers['scaler']
        model = st.session_state.best_model
        chi2_features = st.session_state.chi2_features

        try:
            # 1. Transform Temperature -> Temp_Category
            temp_cat = kbins.transform([[temp]])[0][0]
            
            # 2. Normalize 3 columns
            norm_vals = scaler.transform([[temp, precip, co2]])[0]
            norm_temp, norm_precip, norm_co2 = norm_vals[0], norm_vals[1], norm_vals[2]
            
            # 3. Log Transform Economic Impact
            econ_log = np.log1p(economic)

            # Buat dictionary input
            input_dict = {
                'Year': year,
                'Average_Temperature_C': norm_temp,
                'Total_Precipitation_mm': norm_precip,
                'CO2_Emissions_MT': norm_co2,
                'Economic_Impact_Million_USD': economic,
                'Economic_Impact_Log': econ_log,
                'Temp_Category': temp_cat
            }
            
            # Buat DataFrame hanya dengan kolom chi2_features sesuai urutan
            input_df = pd.DataFrame([input_dict])[chi2_features]
            
            # Lakukan Prediksi dan dapatkan Probabilitas
            with st.spinner("Menganalisa data..."):
                prediction = model.predict(input_df)[0]
                probs = model.predict_proba(input_df)[0]
                classes = model.classes_
            
            # Mapping warna untuk hasil (Tabler Colors)
            color_map = {
                'Rendah': '#d63939',  # Tabler Danger Red
                'Sedang': '#f76707',  # Tabler Warning Orange
                'Tinggi': '#2fb344'   # Tabler Success Green
            }
            color = color_map.get(prediction, '#206bc4')

            # Menampilkan Kotak Hasil
            st.markdown(f"""
            <div class="pred-box">
                <h4>Prediksi Kelas Hasil Panen</h4>
                <div class="pred-text" style="color: {color}; font-size: 2.5rem;">{prediction.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Membagi layout untuk grafik dan teks rekomendasi
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.markdown("### <i class='fa-solid fa-chart-bar'></i> Tingkat Keyakinan Model", unsafe_allow_html=True)
                import plotly.express as px
                prob_df = pd.DataFrame({
                    'Kelas': classes,
                    'Probabilitas (%)': probs * 100
                }).sort_values('Probabilitas (%)')
                
                fig = px.bar(prob_df, x='Probabilitas (%)', y='Kelas', orientation='h',
                             color='Kelas', color_discrete_map=color_map,
                             text='Probabilitas (%)')
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='auto')
                fig.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
            with res_col2:
                st.markdown("### <i class='fa-solid fa-clipboard-list'></i> Analisis & Rekomendasi", unsafe_allow_html=True)
                if prediction == 'Rendah':
                    st.markdown("<div class='success-box' style='background: #fdf3f3; border-left-color: #d63939;'><b>Tindakan Diperlukan!</b> Cuaca ekstrem, emisi tinggi, atau suhu yang kurang ideal mungkin menjadi penyebab utama prediksi panen rendah. Disarankan untuk segera menerapkan strategi adaptasi seperti penggunaan bibit tahan cuaca atau optimalisasi irigasi.</div>", unsafe_allow_html=True)
                elif prediction == 'Sedang':
                    st.markdown("<div class='success-box' style='background: #fdf8f3; border-left-color: #f76707;'><b>Kondisi Cukup Stabil.</b> Potensi panen berada di tingkat menengah. Hasil masih bisa dimaksimalkan melalui perbaikan sistem pemupukan yang lebih efisien dan kontrol kualitas tanah.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='success-box'><b>Kondisi Sangat Ideal!</b> Iklim dan faktor lingkungan sangat mendukung untuk hasil panen yang maksimal. Pertahankan pola dan strategi pertanian yang sedang digunakan saat ini.</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses input: {str(e)}")
