with open('C:/Semester6/Teori/Data Mining/Dashboard/templates/index.html','r',encoding='utf-8') as f:
    lines = f.readlines()

keywords = ['Parameter', 'has_result', 'col-12 col-lg', 'form-footer', 'html_gauge', 'html_scatter', 'lottie', 'prediction_result', 'Skenario', 'Peringatan Bencana', 'Indikator', 'row-deck', 'row-cards', 'Jalankan Prediksi', 'empty-title', 'bg-success-lt', 'bg-danger-lt', 'confidence']
for i, l in enumerate(lines):
    s = l.strip()
    for kw in keywords:
        if kw in s:
            print(f'{i+1}: {s[:150]}')
            break
