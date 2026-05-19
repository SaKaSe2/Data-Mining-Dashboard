import sys
import re

with open('C:/Semester6/Teori/Data Mining/tabler_demo.html', 'r', encoding='utf-8') as f:
    html = f.read()

# We will inject our form block at the beginning
form_html = '''
<!-- Form Panel -->
<div class="col-12 col-lg-4">
    <div class="card">
    <div class="card-header">
        <h3 class="card-title">Parameter Input Iklim & Ekonomi</h3>
    </div>
    <div class="card-body">
        <form action="/predict" method="POST">
        <div class="mb-3">
            <label class="form-label required">Average Temperature (°C)</label>
            <div>
            <input type="number" step="0.01" class="form-control" name="temp" value="{{ temp|default('25.5') }}" placeholder="Contoh: 28.5" required>
            <small class="form-hint">Suhu rata-rata di area observasi.</small>
            </div>
        </div>
        <div class="mb-3">
            <label class="form-label required">Total Precipitation (mm)</label>
            <div>
            <input type="number" step="0.01" class="form-control" name="precip" value="{{ precip|default('1200.0') }}" placeholder="Contoh: 1500" required>
            </div>
        </div>
        <div class="mb-3">
            <label class="form-label required">CO2 Emissions (MT)</label>
            <div>
            <input type="number" step="0.01" class="form-control" name="co2" value="{{ co2|default('50.0') }}" required>
            </div>
        </div>
        <div class="mb-4">
            <label class="form-label required">Economic Impact (Million USD)</label>
            <div>
            <div class="input-group">
                <span class="input-group-text">$</span>
                <input type="number" step="0.01" class="form-control" name="economic" value="{{ economic|default('500.0') }}" required>
            </div>
            </div>
        </div>
        <div class="form-footer">
            <button type="submit" class="btn btn-primary w-100">Jalankan Prediksi AI</button>
        </div>
        </form>
    </div>
    </div>
</div>

<!-- Result Area -->
<div class="col-12 col-lg-8">
    {% if has_result %}
        {% if prediction_result == 'Tinggi' %}
        <div class="alert alert-success alert-important" role="alert">
        <div class="d-flex">
            <div>
            <svg xmlns="http://www.w3.org/2000/svg" class="icon alert-icon" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M5 12l5 5l10 -10" /></svg>
            </div>
            <div>
            <h4 class="alert-title">Skenario Optimal: Hasil Panen Diprediksi TINGGI</h4>
            <div class="text-secondary mt-1">Kondisi saat ini sangat mendukung peningkatan yield. Prediksi confidence level dari AI adalah {{ confidence }}.</div>
            </div>
        </div>
        </div>
        {% else %}
        <div class="alert alert-danger alert-important" role="alert">
        <div class="d-flex">
            <div>
            <svg xmlns="http://www.w3.org/2000/svg" class="icon alert-icon" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0 -18 0" /><path d="M12 8l0 4" /><path d="M12 16l.01 0" /></svg>
            </div>
            <div>
            <h4 class="alert-title">Peringatan: Hasil Panen Diprediksi RENDAH</h4>
            <div class="text-secondary mt-1">Risiko tinggi akan penurunan drastis pada yield crop akibat parameter iklim ekstrim. Prediksi confidence level dari AI adalah {{ confidence }}.</div>
            </div>
        </div>
        </div>
        {% endif %}

        <div class="row row-cards mt-2">
        <div class="col-sm-6">
            <div class="card">
            <div class="card-header">
                <h3 class="card-title">Confidence Indicator</h3>
            </div>
            <div class="card-body p-0">
                {{ html_gauge|safe }}
            </div>
            </div>
        </div>
        <div class="col-sm-6">
            <div class="card">
            <div class="card-header">
                <h3 class="card-title">Cuaca Anda vs Historis</h3>
            </div>
            <div class="card-body p-0">
                {{ html_scatter|safe }}
            </div>
            </div>
        </div>
        </div>

    {% else %}
    <div class="card d-flex flex-column h-100 align-items-center justify-content-center p-5">
    <div class="empty">
        <div class="empty-img" id="lottie-animation" style="max-width: 300px; margin: 0 auto;"></div>
        <p class="empty-title">Menunggu Parameter Iklim</p>
        <p class="empty-subtitle text-secondary">
        Masukkan data observasi di sebelah kiri untuk melihat prediksi performa hasil panen dari model kecerdasan buatan.
        </p>
    </div>
    </div>
    
    <script>
    document.addEventListener('DOMContentLoaded', function () {
        var animation = bodymovin.loadAnimation({
            container: document.getElementById('lottie-animation'),
            renderer: 'svg',
            loop: true,
            autoplay: true,
            path: 'https://assets2.lottiefiles.com/packages/lf20_q5pk6p1k.json' 
        });
    });
    </script>
    {% endif %}
</div>
'''

# Use simple string replacement to insert our form after the first row-cards
target_str = '<div class="row row-deck row-cards">'
if target_str in html:
    html = html.replace(target_str, target_str + '\n' + form_html, 1)

# Correct paths to use the local tabler server which is where they downloaded it, or public CDN
html = html.replace('href="./dist/', 'href="https://cdn.jsdelivr.net/npm/@tabler/core@latest/dist/')
html = html.replace('src="./dist/', 'src="https://cdn.jsdelivr.net/npm/@tabler/core@latest/dist/')

# The preview demo css is local to their machine (Tabler repo), we can mount it in main.py or just leave it. 
# They ran 'pnpm dev' so we can point to their localhost!
html = html.replace('href="./preview/', 'href="http://localhost:3000/preview/')
html = html.replace('src="./preview/', 'src="http://localhost:3000/preview/')

# Add lottie and plotly to head
head_closing = '</head>'
scripts_to_add = '<script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js"></script><script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>\n'
if head_closing in html:
    html = html.replace(head_closing, scripts_to_add + head_closing)

# Replace <title>
html = re.sub(r'<title>.*?</title>', '<title>{{ title }} - Data Mining Dashboard</title>', html, flags=re.DOTALL)

with open('C:/Semester6/Teori/Data Mining/Dashboard/templates/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
print("Template replaced!")
