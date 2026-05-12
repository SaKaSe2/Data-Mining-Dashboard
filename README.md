# Dashboard Data Mining - Climate Change Impact on Agriculture

Dashboard interaktif untuk analisis Data Mining menggunakan algoritma **Random Forest Classification** pada dataset Climate Change Impact on Agriculture 2024.

## Persyaratan

- Python 3.9 atau lebih baru
- pip (package manager Python)

## Instalasi

1. Buka terminal / command prompt

2. Masuk ke folder Dashboard:
   ```
   cd C:\Semester6\Teori\Data Mining\Dashboard
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Cara Menjalankan

```
cd C:\Semester6\Teori\Data Mining\Dashboard
streamlit run app.py
```

Dashboard akan terbuka otomatis di browser pada alamat `http://localhost:8501`.

## Alur Penggunaan

Dashboard memiliki 5 halaman yang harus dijalankan secara berurutan melalui sidebar di sebelah kiri:

### 1. Overview Dataset
- Menampilkan preview data, statistik deskriptif, dan informasi kolom.
- Halaman ini otomatis tampil saat pertama kali dibuka, tidak perlu klik apapun.

### 2. Preprocessing
- Klik menu **"2. Preprocessing"** di sidebar.
- Klik tombol **"Jalankan Preprocessing"**.
- Tunggu hingga muncul pesan "Preprocessing selesai!".
- Halaman ini menjalankan seluruh pipeline: Data Cleaning, Integration, Transformation, PCA, Discretization, Normalization, Feature Selection, dan TF-IDF.

### 3. Feature Selection
- Klik menu **"3. Feature Selection"** di sidebar.
- Pada tab **Chi-Square**, klik **"Jalankan Chi-Square Selection"** dan tunggu selesai.
- Pindah ke tab **RFE**, klik **"Jalankan RFE Selection"** dan tunggu selesai.
- Kedua metode ini harus dijalankan sebelum lanjut ke halaman berikutnya.

### 4. Eksperimen Model
- Klik menu **"4. Eksperimen Model"** di sidebar.
- Klik tombol **"Jalankan Semua Eksperimen"**.
- Proses ini membutuhkan waktu sekitar 3-5 menit karena menjalankan GridSearchCV.
- Tunggu hingga progress bar mencapai 100% dan muncul pesan sukses.
- Setelah selesai, expand setiap eksperimen untuk melihat detail metrik, confusion matrix, dan feature importance.

### 5. Perbandingan Hasil
- Klik menu **"5. Perbandingan Hasil"** di sidebar.
- Menampilkan tabel perbandingan seluruh eksperimen, chart perbandingan, dan kesimpulan.
- Daftar file output yang tersimpan di folder `data/` juga ditampilkan di sini.

## Output

Semua hasil visualisasi dan data tersimpan di folder `Dashboard/data/`:

| File | Keterangan |
|---|---|
| `comparison_table.csv` | Tabel perbandingan metrik 4 eksperimen |
| `comparison_chart.png` | Bar chart perbandingan Accuracy, Precision, Recall, F1 |
| `chi2_scores.png` | Skor Chi-Square per fitur |
| `rfe_ranking.png` | Ranking RFE per fitur |
| `distribution_target.png` | Distribusi nilai Crop Yield (target) |
| `class_distribution.png` | Distribusi kelas Rendah/Sedang/Tinggi |
| `confusion_matrix_eks1.png` | Confusion matrix eksperimen 1 (Baseline) |
| `confusion_matrix_eks2.png` | Confusion matrix eksperimen 2 (Chi-Square) |
| `confusion_matrix_eks3.png` | Confusion matrix eksperimen 3 (RFE) |
| `confusion_matrix_eks4.png` | Confusion matrix eksperimen 4 (Tuned) |
| `feature_importance_eks1.png` | Feature importance eksperimen 1 |
| `feature_importance_eks2.png` | Feature importance eksperimen 2 |
| `feature_importance_eks3.png` | Feature importance eksperimen 3 |
| `feature_importance_eks4.png` | Feature importance eksperimen 4 |

## Struktur Folder

```
Dashboard/
├── app.py              # Aplikasi Streamlit (UI dashboard)
├── pipeline.py         # Modul pipeline preprocessing dan modeling
├── requirements.txt    # Daftar dependencies
├── README.md           # Dokumentasi ini
└── data/               # Folder output hasil model
```

## Detail Teknis

- **Dataset**: `climate_change_impact_on_agriculture_2024 (1).csv` (10.000 baris, 15 kolom)
- **Algoritma**: Random Forest Classification
- **Split Data**: Train 80% / Test 20%
- **Cross-Validation**: 5-Fold Stratified
- **Target**: `Crop_Yield_MT_per_HA` didiskritisasi menjadi 3 kelas (Rendah, Sedang, Tinggi)
- **Seleksi Fitur**: Chi-Square (top 7) dan RFE (top 7)
- **Hyperparameter Tuning**: GridSearchCV (225 kombinasi)
