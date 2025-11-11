# Web Scraping + Machine Learning (Regresión de precios)

Proyecto de nivel avanzado que combina scraping web real con un pipeline completo de ML para predecir el precio de productos.

- Fuente de datos: Books to Scrape (https://books.toscrape.com/) — sitio de prueba apto para scraping.
- Objetivo: Predecir el precio de un libro en función de características numéricas y categóricas.
- Modelos: Baseline (RandomForestRegressor) y una red neuronal densa (Keras) [64-128-64-1].
- Métricas: R² y MAE con split 80/20.

## Estructura

```
/data
/src
  scraper.py
  train_model.py
  evaluate.py
/notebooks
  EDA_and_Training.ipynb
/models
/results
requirements.txt
README.md
```

## Requisitos

- Python 3.10+
- Librerías clave: requests, beautifulsoup4, pandas, numpy, scikit-learn, tensorflow, matplotlib, seaborn

Instala dependencias:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uso

1) Scraping para construir el dataset (1000+ registros):
```bash
python src/scraper.py --out data/dataset.csv
```
Opcionales: `--max-items`, `--min-delay`, `--max-delay`.

2) Entrenamiento (baseline + red neuronal) y guardado de artefactos:
```bash
python src/train_model.py --data data/dataset.csv
```
Artefactos generados:
- models/model.h5 (red neuronal)
- models/baseline_random_forest.joblib
- models/preprocessor.joblib
- models/feature_names.json
- results/metrics.json
- results/predictions.csv
- results/history_nn.json

3) Evaluación y gráficas:
```bash
python src/evaluate.py
```
Gráficas en /results:
- scatter_baseline.png, scatter_nn.png
- residuals_baseline.png, residuals_nn.png
- feature_importances_top20.png
- training_history.png

## Notebook (EDA + Entrenamiento)
Abre `notebooks/EDA_and_Training.ipynb` para un flujo interactivo de análisis y entrenamiento.

## Notas sobre scraping
- Books to Scrape es un sitio de demostración, pensado para prácticas de scraping.
- Se respeta un retardo breve entre peticiones para ser cortés (configurable por CLI).

## Reproducibilidad
- Se fijan semillas aleatorias (numpy/tensorflow/sklearn) para mayor estabilidad de resultados.
- El split es 80/20 estricto para evaluación justa.

## Troubleshooting
- En Windows, asegúrate de activar el entorno virtual antes de instalar dependencias.
- TensorFlow CPU 2.15 funciona en Python 3.10; si tienes GPU/CUDA, ajusta la instalación según tu entorno.
