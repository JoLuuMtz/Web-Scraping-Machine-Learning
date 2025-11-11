# Explain: Funcionamiento del proyecto (de principio a fin)

Este documento explica, de forma práctica y técnica, cómo funciona el proyecto de Web Scraping + Machine Learning, qué librerías utiliza, cómo se conectan sus componentes, qué contiene cada carpeta y cuál es el flujo completo de extremo a extremo.

---

## 1) Resumen ejecutivo

- Se extraen datos reales desde el sitio de prueba Books to Scrape (https://books.toscrape.com/).
- Se construye un dataset con 1000+ filas con campos numéricos y categóricos.
- Se entrena un modelo baseline (RandomForestRegressor) y un modelo avanzado (Red Neuronal Densa – Keras).
- Se evalúa con R² y MAE (split 80/20) y se generan métricas, predicciones y gráficas.
- Todo el flujo puede ejecutarse por CLI o desde un notebook con EDA.

---

## 2) Arquitectura y flujo end‑to‑end

```
requests + BeautifulSoup  --->  src/scraper.py  --->  data/dataset.csv
                                     |
                                     v
                         src/train_model.py (pandas, scikit-learn, tensorflow)
                                     |
                          +-----------+-----------+
                          |                       |
                          v                       v
               models/preprocessor.joblib     models/model.h5
               models/baseline_random_...     results/history_nn.json
               models/feature_names.json      results/predictions.csv
                                             results/metrics.json
                                     |
                                     v
                              src/evaluate.py (matplotlib, seaborn)
                                     |
                                     v
                               results/*.png (gráficas)
```

- `src/scraper.py` recolecta y limpia datos; guarda `data/dataset.csv`.
- `src/train_model.py` carga el dataset, preprocesa features y entrena ambos modelos; guarda artefactos en `models/` y `results/`.
- `src/evaluate.py` carga predicciones y métricas, y genera gráficas de evaluación en `results/`.
- `notebooks/EDA_and_Training.ipynb` permite ejecutar todo de forma interactiva (scraping, EDA, entrenamiento, evaluación).

---

## 3) Estructura de carpetas y contenido

- `/data`
  - `dataset.csv`: dataset consolidado tras scraping y limpieza.

- `/src`
  - `scraper.py`: scraping del sitio (categorías, páginas, productos) y armado del dataset.
  - `train_model.py`: preprocesamiento, entrenamiento (RandomForest y NN), guardado de artefactos y métricas.
  - `evaluate.py`: generación de gráficas (dispersión, residuales, importancias, historia de entrenamiento).

- `/models`
  - `model.h5`: red neuronal Keras entrenada (formato HDF5).
  - `baseline_random_forest.joblib`: modelo baseline RandomForest.
  - `preprocessor.joblib`: transformador (ColumnTransformer) con imputaciones, escalado y one‑hot.
  - `feature_names.json`: nombres de features resultantes tras el preprocesamiento.

- `/results`
  - `metrics.json`: métricas finales (R², MAE) para baseline y NN.
  - `predictions.csv`: y_true, y_pred_baseline, y_pred_nn (partición de test).
  - `history_nn.json`: historial de pérdida (loss/val_loss) por época para la NN.
  - `*.png`: gráficas (dispersión, residuales, importancias, historia de entrenamiento).

- `/notebooks`
  - `EDA_and_Training.ipynb`: notebook con celdas para scraping, EDA, entrenamiento y evaluación.

- Archivos raíz
  - `requirements.txt`: dependencias del proyecto.
  - `README.md`: guía rápida de instalación y uso.
  - `Explain.md`: este documento explicativo.

---

## 4) Datos y features

De cada libro se extraen, entre otros, los siguientes campos:

- `title` (texto)
- `price` (float; variable objetivo para regresión)
- `category` (categórica)
- `rating` (numérica; 1 a 5)
- `availability` (número de unidades disponibles)
- `description_len` (longitud de la descripción)
- `has_desc` (binaria; 0/1 si hay descripción)
- `n_reviews` (número de reseñas)
- `title_len` (longitud del título)
- `url` (enlace del producto)

Limpieza mínima: se eliminan registros con nulos en `price`, `rating` o `category`.

---

## 5) Preprocesamiento de features (scikit‑learn)

`ColumnTransformer` aplica diferentes transformaciones por tipo de variable:

- Numéricas (`rating`, `availability`, `description_len`, `title_len`, `n_reviews`, `has_desc`)
  - `SimpleImputer(strategy="median")`
  - `StandardScaler(with_mean=True)`

- Categóricas (`category`)
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")` (soporta categorías no vistas)

Se guardan los nombres de columnas transformadas en `models/feature_names.json` para trazabilidad.

---

## 6) Modelos y entrenamiento

- Baseline: `RandomForestRegressor`
  - `n_estimators=300`, `n_jobs=-1`, `random_state=42`.

- Avanzado: Red Neuronal (Keras Sequential)
  - Capas: `[64, 128, 64, 1]` con activación `ReLU` en ocultas.
  - Optimizador: `Adam(learning_rate=1e-3)`.
  - Pérdida: `MSE`.
  - `EarlyStopping(patience=15, monitor="val_loss", restore_best_weights=True)`.
  - Entrenamiento con `validation_split=0.2`.

- División de datos: `train_test_split(test_size=0.2, random_state=42)`.
- Semillas fijadas para reproducibilidad (NumPy/TensorFlow/Python).

Artefactos resultantes:

- Modelos: `models/model.h5`, `models/baseline_random_forest.joblib`.
- Preprocesador: `models/preprocessor.joblib`.
- Nombres de features: `models/feature_names.json`.
- Predicciones (test): `results/predictions.csv`.
- Métricas: `results/metrics.json`.
- Historia NN: `results/history_nn.json`.

---

## 7) Evaluación y gráficas (matplotlib + seaborn)

- Dispersión `y_true` vs `y_pred` para baseline y NN: `scatter_*.png`.
- Histograma de residuales (pred − true): `residuals_*.png`.
- Importancias de features (RandomForest): `feature_importances_top20.png`.
- Historia de entrenamiento de la NN (loss y val_loss por epoch): `training_history.png`.

Métricas de referencia: `R²` y `MAE` en `results/metrics.json` y en salida de consola.

---

## 8) Ejecución (CLI)

1. Crear/activar entorno (Windows PowerShell)
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Scraping (genera `data/dataset.csv`)
   ```powershell
   python src/scraper.py --out data/dataset.csv --min-delay 0.2 --max-delay 0.5
   # Opcionales: --max-items, --min-delay, --max-delay
   ```

3. Entrenamiento (genera modelos y métricas)
   ```powershell
   python src/train_model.py --data data/dataset.csv --epochs 100 --batch-size 32 --rf-estimators 300
   ```

4. Evaluación (genera gráficas en `results/`)
   ```powershell
   python src/evaluate.py
   ```

Alternativa: abrir y ejecutar `notebooks/EDA_and_Training.ipynb`.

---

## 9) Dependencias clave (mapa rápido)

- `requests`, `beautifulsoup4`: Scraping HTTP/HTML (`src/scraper.py`).
- `pandas`, `numpy`: Carga y manipulación de datos.
- `scikit-learn`: Preprocesamiento (imputación, escalado, OHE), split, RandomForest, métricas.
- `tensorflow/keras`: Red neuronal densa para regresión.
- `matplotlib`, `seaborn`: Gráficas de evaluación.
- `joblib`: Serialización de modelos/preprocesadores.
- `tqdm` (opcional): Barras de progreso si se incorporan.

---

## 10) Manejo de errores y cortesía en scraping

- Peticiones robustas con `session.get` y `raise_for_status()`.
- Si una página de producto falla al parsear, se omite (retorna `None`).
- Se aplica retardo aleatorio entre peticiones (`--min-delay`, `--max-delay`) para ser cortés con el sitio.
- Se evita duplicar productos mediante un conjunto `seen` de URLs ya procesadas.

---

## 11) Reproducibilidad y consideraciones

- Semillas fijas (`numpy`, `random`, `tensorflow`) para estabilidad.
- Split 80/20 dedicado a evaluación justa.
- Formato de modelos: Keras `.h5` (legacy pero ampliamente compatible) y `joblib` para scikit‑learn.

---

## 12) Limitaciones y mejoras sugeridas

- El conjunto Books to Scrape puede no tener alta señal predictiva en precio con las features básicas, resultando en R² bajos.
- Posibles mejoras:
  - Features de texto (TF‑IDF de `title`/`description`) + reducción dimensional (TruncatedSVD).
  - Transformación del objetivo (`log(price)`) y re‑entrenamiento.
  - Modelos adicionales (HistGradientBoosting, XGBoost/LightGBM) y búsqueda de hiperparámetros.
  - Conversión a clasificación por rangos de precio.

---

## 13) Trazabilidad de artefactos

- `models/`
  - `model.h5`, `baseline_random_forest.joblib`, `preprocessor.joblib`, `feature_names.json`.
- `results/`
  - `metrics.json`, `predictions.csv`, `history_nn.json`, `*.png`.
- `data/`
  - `dataset.csv` (fuente para entrenamiento y evaluación).

Con esto, dispones de una visión completa del proyecto, sus componentes y sus conexiones, junto con las rutas y comandos necesarios para reproducir el flujo de punta a punta.
