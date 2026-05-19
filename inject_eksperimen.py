import sys
import re

with open('C:/Semester6/Teori/Data Mining/tabler_demo.html', 'r', encoding='utf-8') as f:
    html = f.read()

# We will inject our eksperimen block at the beginning
form_html = '''
<!-- Stat Cards -->
<div class="col-sm-6 col-lg-3">
<div class="card card-sm">
    <div class="card-body">
    <div class="row align-items-center">
        <div class="col-auto">
        <span class="bg-primary text-white avatar">
            <svg xmlns="http://www.w3.org/2000/svg" class="icon" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M5 12l5 5l10 -10" /></svg>
        </span>
        </div>
        <div class="col">
        <div class="font-weight-medium">Akurasi</div>
        <div class="text-secondary">~84.5%</div>
        </div>
    </div>
    </div>
</div>
</div>

<div class="col-sm-6 col-lg-3">
<div class="card card-sm">
    <div class="card-body">
    <div class="row align-items-center">
        <div class="col-auto">
        <span class="bg-green text-white avatar">
            <svg xmlns="http://www.w3.org/2000/svg" class="icon" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0 -18 0" /><path d="M12 12l3 2" /><path d="M12 7v5" /></svg>
        </span>
        </div>
        <div class="col">
        <div class="font-weight-medium">Precision</div>
        <div class="text-secondary">~85.2%</div>
        </div>
    </div>
    </div>
</div>
</div>

<div class="col-sm-6 col-lg-3">
<div class="card card-sm">
    <div class="card-body">
    <div class="row align-items-center">
        <div class="col-auto">
        <span class="bg-orange text-white avatar">
            <svg xmlns="http://www.w3.org/2000/svg" class="icon" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M3 12a9 9 0 1 0 18 0a9 9 0 0 0 -18 0" /><path d="M12 7v5l3 3" /></svg>
        </span>
        </div>
        <div class="col">
        <div class="font-weight-medium">Recall</div>
        <div class="text-secondary">~83.9%</div>
        </div>
    </div>
    </div>
</div>
</div>

<div class="col-sm-6 col-lg-3">
<div class="card card-sm">
    <div class="card-body">
    <div class="row align-items-center">
        <div class="col-auto">
        <span class="bg-purple text-white avatar">
            <svg xmlns="http://www.w3.org/2000/svg" class="icon" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0 -18 0" /><path d="M12 12l4 4" /><path d="M12 12l-4 -4" /></svg>
        </span>
        </div>
        <div class="col">
        <div class="font-weight-medium">F1-Score</div>
        <div class="text-secondary">~84.5%</div>
        </div>
    </div>
    </div>
</div>
</div>

<!-- Visualizations -->
<div class="col-md-6">
<div class="card">
    <div class="card-header"><h3 class="card-title">Feature Importance (Random Forest)</h3></div>
    <div class="card-body p-0 text-center bg-white">
    <img src="/data/feature_importance_eks4.png" alt="Feature Importance" class="img-fluid" style="width:100%; border-radius: 0 0 4px 4px;">
    </div>
</div>
</div>

<div class="col-md-6">
<div class="card">
    <div class="card-header"><h3 class="card-title">Confusion Matrix</h3></div>
    <div class="card-body p-0 text-center bg-white">
    <img src="/data/confusion_matrix_eks4.png" alt="Confusion Matrix" class="img-fluid" style="width:100%; border-radius: 0 0 4px 4px;">
    </div>
</div>
</div>

<div class="col-md-12">
<div class="card">
    <div class="card-header"><h3 class="card-title">Chi-Square Feature Selection</h3></div>
    <div class="card-body p-0 text-center bg-white">
    <img src="/data/chi2_scores.png" alt="Chi2 Scores" class="img-fluid" style="width:100%; max-height: 500px; object-fit: contain; border-radius: 0 0 4px 4px;">
    </div>
</div>
</div>
'''

target_str = '<div class="row row-deck row-cards">'
if target_str in html:
    html = html.replace(target_str, target_str + '\n' + form_html, 1)

# Modify header text
html = re.sub(r'<div class="page-pretitle">.*?</div>', '<div class="page-pretitle">Performance Metrics</div>', html, count=1, flags=re.DOTALL)
html = re.sub(r'<h2 class="page-title">.*?</h2>', '<h2 class="page-title">Hasil Evaluasi & Pelatihan Model</h2>', html, count=1, flags=re.DOTALL)

html = html.replace('href="./dist/', 'href="https://cdn.jsdelivr.net/npm/@tabler/core@latest/dist/')
html = html.replace('src="./dist/', 'src="https://cdn.jsdelivr.net/npm/@tabler/core@latest/dist/')
html = html.replace('href="./preview/', 'href="http://localhost:3000/preview/')
html = html.replace('src="./preview/', 'src="http://localhost:3000/preview/')

head_closing = '</head>'
scripts_to_add = '<script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js"></script><script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>\n'
if head_closing in html:
    html = html.replace(head_closing, scripts_to_add + head_closing)

html = re.sub(r'<title>.*?</title>', '<title>{{ title }} - Data Mining Dashboard</title>', html, flags=re.DOTALL)

with open('C:/Semester6/Teori/Data Mining/Dashboard/templates/eksperimen.html', 'w', encoding='utf-8') as f:
    f.write(html)
print("Template replaced!")
