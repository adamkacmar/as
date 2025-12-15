## Architektúra softvéru - jednoduchá API pre modely počítačového videnia a ich merania

### Spustenie
```bash
pip install -r requirements.txt
python swinv2_api.py
```

### Endpointy
- `GET /health` – stav, zariadenie, model_loaded
- `GET /classes` – zoznam tried a počet
- `POST /predict` – form-data `image` (súbor); vráti triedu, confidence, metriky
- `POST /benchmark` – form-data `images` (viac súborov); vráti metriky na obrázok a agregované štatistiky

### Príklady requestov
```bash
curl http://localhost:5000/health
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/predict
curl -X POST -F "images=@img1.jpg" -F "images=@img2.jpg" http://localhost:5000/benchmark
```

### Ouputy
Predict: `predicted_class`, `confidence`, `top_predictions`, `metrics` (preprocessing_time_ms, inference_time_ms, total_time_ms, device).
Benchmark: počty (images, successful, failed), časy na obrázok, `statistics` (avg/min/max inference, avg preprocessing/total, throughput, device).

### Ďalšie súbory
- `test_client.py` – jednoduchý tester endpointov
- `benchmark_example.py` – single/batch benchmarky proti API

### Poznámka
Model nie je zverejnený na GitHub. Bol nátrenový pomocou nástrojov z ďalšieho projektu: https://github.com/adamkacmar/bc-grape-disease

Autori: Adam Kačmár a Martin Hnatko
Rok: 2025