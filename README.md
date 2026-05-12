# Dashboard Data Mining - Climate Change Impact on Agriculture

Dashboard interaktif bergaya profesional (terinspirasi dari **Tabler UI**) untuk analisis Data Mining menggunakan algoritma **Random Forest Classification** pada dataset Climate Change Impact on Agriculture 2024.

## Persyaratan
- Python 3.9 atau lebih baru
- `pip` (package manager Python)

## Instalasi
1. Buka terminal (CMD / PowerShell / VSCode).
2. Masuk ke folder Dashboard:
   ```cmd
   cd "C:\Semester6\Teori\Data Mining\Dashboard"
   ```
3. Install dependencies (Pastikan terkoneksi internet):
   ```cmd
   pip install -r requirements.txt
   ```

## Cara Menjalankan Aplikasi
Setelah instalasi selesai, jalankan perintah berikut di terminal yang sama:
```cmd
streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

---

## Alur Penggunaan Dashboard
Dashboard ini telah disederhanakan menjadi **2 menu utama** untuk pengalaman pengguna yang lebih mulus dan anti-ribet.

### 1. Eksperimen & Hasil
- Menu ini adalah dapur pacu Data Mining.
- Cukup dengan satu kali klik **"Jalankan Ulang Semua Proses"**, sistem akan otomatis memproses 4 tahap krusial secara aman (Anti-Freeze / Deadlock di Windows):
  1. Preprocessing Data & Transformasi.
  2. Feature Selection (Chi-Square & RFE).
  3. Evaluasi Model (Baseline, Chi-Square, RFE).
  4. Hyperparameter Tuning (GridSearchCV).
- Hasil terbaik akan otomatis disimpan ke dalam file `.pkl` agar bisa langsung digunakan.
- Tersedia tabel komparasi dan visualisasi bar chart performa di bagian bawah.

### 2. Simulasi Prediksi (Uji Model)
- Menu ini memungkinkan kamu mengetes model AI layaknya aplikasi sungguhan.
- **Instant Load**: Karena model sudah disimpan dalam `.pkl`, kamu tidak perlu menunggu proses *training* lagi.
- Masukkan parameter cuaca (Suhu, Hujan, Emisi) dan Dampak Ekonomi.
- AI akan memberikan Prediksi Hasil Panen (**Rendah / Sedang / Tinggi**), menampilkan grafik probabilitas tingkat keyakinan (Confidence Level), dan memberikan rekomendasi strategis secara langsung.

---

## Output File
Semua hasil visualisasi dan model AI akan tersimpan di folder `Dashboard/data/`:
- `best_model.pkl`: Model AI yang sudah jadi.
- `transformers.pkl` & `chi2_features.pkl`: Logik normalisasi data.
- Gambar `.png`: Menyimpan visualisasi akurasi, _confusion matrix_, dan _feature importance_.

## Catatan Penting Terkait Prediksi
Algoritma Random Forest yang dilatih pada dataset ini memberikan bobot paling besar pada fitur **Dampak Ekonomi (Economic Impact)** (mencapai ~60% *Feature Importance*). Oleh karena itu, saat mencoba simulasi, perubahan ekstrem pada cuaca tidak akan terlalu berdampak jika nilai Dampak Ekonomi tidak kamu ubah secara signifikan.
