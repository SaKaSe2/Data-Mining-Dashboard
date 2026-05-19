import re

with open('C:/Semester6/Teori/Data Mining/Dashboard/templates/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# We want to replace the two alert divs inside {% if prediction_result == 'Tinggi' %} and {% else %}
# The existing pattern looks like this:
old_high = '''<div class="alert alert-success alert-important" role="alert">
        <div class="d-flex">
            <div>
            <svg xmlns="http://www.w3.org/2000/svg" class="icon alert-icon" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M5 12l5 5l10 -10" /></svg>
            </div>
            <div>
            <h4 class="alert-title">Skenario Optimal: Hasil Panen Diprediksi TINGGI</h4>
            <div class="text-secondary mt-1">Kondisi saat ini sangat mendukung peningkatan yield. Prediksi confidence level dari AI adalah {{ confidence }}.</div>
            </div>
        </div>
        </div>'''

old_low = '''<div class="alert alert-danger alert-important" role="alert">
        <div class="d-flex">
            <div>
            <svg xmlns="http://www.w3.org/2000/svg" class="icon alert-icon" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0 -18 0" /><path d="M12 8l0 4" /><path d="M12 16l.01 0" /></svg>
            </div>
            <div>
            <h4 class="alert-title">Peringatan: Hasil Panen Diprediksi RENDAH</h4>
            <div class="text-secondary mt-1">Risiko tinggi akan penurunan drastis pada yield crop akibat parameter iklim ekstrim. Prediksi confidence level dari AI adalah {{ confidence }}.</div>
            </div>
        </div>
        </div>'''

new_high = '''<div class="card bg-success-lt mb-3">
  <div class="card-body text-center p-5">
    <div class="mb-4">
      <img src="https://media.giphy.com/media/l41YkxvU8c7J7Bba0/giphy.gif" style="width: 250px; height: 250px; object-fit: cover; border-radius: 50%; box-shadow: 0 4px 15px rgba(0,0,0,0.1);" alt="Musim Semi Happy">
    </div>
    <h1 class="text-success fw-bold">Skenario Optimal: Panen TINGGI!</h1>
    <p class="text-secondary fs-3">Musim semi yang cerah! Kondisi saat ini sangat mendukung peningkatan hasil panen secara optimal.</p>
    <div class="mt-4">
      <span class="badge bg-success text-white px-3 py-2 fs-4">Tingkat Kepercayaan: {{ confidence }}</span>
    </div>
  </div>
</div>'''

new_low = '''<div class="card bg-danger-lt mb-3">
  <div class="card-body text-center p-5">
    <div class="mb-4">
      <img src="https://media.giphy.com/media/26BGD4XaoPO3zTz9K/giphy.gif" style="width: 250px; height: 250px; object-fit: cover; border-radius: 50%; box-shadow: 0 4px 15px rgba(0,0,0,0.1);" alt="Bencana Cuaca Ekstrem">
    </div>
    <h1 class="text-danger fw-bold">Peringatan Bencana: Panen RENDAH!</h1>
    <p class="text-secondary fs-3">Awas cuaca ekstrem! Parameter iklim menunjukkan anomali (badai/salju) yang berpotensi merusak hasil panen secara masif.</p>
    <div class="mt-4">
      <span class="badge bg-danger text-white px-3 py-2 fs-4">Tingkat Kepercayaan: {{ confidence }}</span>
    </div>
  </div>
</div>'''

if old_high in html and old_low in html:
    html = html.replace(old_high, new_high)
    html = html.replace(old_low, new_low)
    with open('C:/Semester6/Teori/Data Mining/Dashboard/templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("Animation cards injected successfully!")
else:
    print("Could not find the exact HTML blocks to replace. They might have changed or spacing is different.")
