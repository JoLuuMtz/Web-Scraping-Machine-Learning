import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _make_ohe() -> OneHotEncoder:
    """Crea un OneHotEncoder compatible con distintas versiones de scikit-learn.
    Returns:
        OneHotEncoder con handle_unknown='ignore' y salida densa (no dispersa).
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Compatibilidad con scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(cat_features: List[str], num_features: List[str]) -> ColumnTransformer:
    """Construye el preprocesador de features basado en ColumnTransformer.
    - Numéricas: imputación por mediana + estandarización estándar.
    - Categóricas: imputación por modo + one-hot encoding (ignorando categorías no vistas).
    Args:
        cat_features: lista de nombres de columnas categóricas.
        num_features: lista de nombres de columnas numéricas.
    Returns:
        ColumnTransformer listo para ajustar/transformar.
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True)),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", _make_ohe()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_features),
            ("cat", cat_pipe, cat_features),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, cat_features: List[str], num_features: List[str]) -> List[str]:
    """Obtiene los nombres de las columnas resultantes tras el preprocesamiento.
    Usa get_feature_names_out cuando está disponible, si no construye los nombres
    concatenando numéricas + categorías one-hot.
    """
    try:
        # scikit-learn >= 1.0
        names = list(preprocessor.get_feature_names_out())
        return [str(n) for n in names]
    except Exception:
        # Fallback: concatenar nombres manuales
        ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        try:
            cat_names = list(ohe.get_feature_names_out(cat_features))
        except Exception:
            cat_names = []
        return list(num_features) + cat_names


def build_nn(input_dim: int) -> keras.Model:
    """Crea y compila una red neuronal densa para regresión.
    Arquitectura: 64-128-64-1, activación ReLU, optimizador Adam (1e-3), pérdida MSE.
    Args:
        input_dim: número de features de entrada tras el preprocesamiento.
    Returns:
        Modelo Keras compilado, listo para entrenar.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def train_and_evaluate(data_path: str = os.path.join("data", "dataset.csv"),
                       epochs: int = 100,
                       batch_size: int = 32,
                       rf_estimators: int = 300,
                       random_state: int = 42) -> Dict:
    """Entrena un modelo baseline (RandomForest) y una red neuronal (Keras) y guarda artefactos.
    Pasos:
      - Carga dataset y define features numéricas y categóricas.
      - Split 80/20 (train/test).
      - Preprocesa con imputación, escalado y one-hot encoding.
      - Entrena RandomForest y NN (con early stopping) y calcula R² y MAE.
      - Guarda modelos, preprocesador, nombres de features, predicciones y métricas.
    Args:
      data_path: ruta al CSV con los datos.
      epochs: épocas para la NN.
      batch_size: tamaño de batch para la NN.
      rf_estimators: número de árboles del RandomForest.
      random_state: semilla de aleatoriedad.
    Returns:
      Diccionario con métricas y metadatos del entrenamiento/evaluación.
    """
    # Semillas
    np.random.seed(random_state)
    random.seed(random_state)
    tf.random.set_seed(random_state)

    # Cargar datos
    df = pd.read_csv(data_path)

    target = "price"
    num_features = ["rating", "availability", "description_len", "title_len", "n_reviews", "has_desc"]
    cat_features = ["category"]
    feature_cols = num_features + cat_features

    df = df.dropna(subset=[target]).reset_index(drop=True)

    X = df[feature_cols]
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    preprocessor = build_preprocessor(cat_features, num_features)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor, cat_features, num_features)

    # Baseline: RandomForestRegressor
    rf = RandomForestRegressor(
        n_estimators=rf_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train_proc, y_train)
    y_pred_rf = rf.predict(X_test_proc)

    # Red Neuronal (Keras)
    model = build_nn(input_dim=X_train_proc.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
    ]
    history = model.fit(
        X_train_proc,
        y_train.values if hasattr(y_train, "values") else y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    y_pred_nn = model.predict(X_test_proc, verbose=0).reshape(-1)

    # Métricas
    metrics = {
        "baseline_random_forest": {
            "r2": float(r2_score(y_test, y_pred_rf)),
            "mae": float(mean_absolute_error(y_test, y_pred_rf)),
        },
        "neural_network": {
            "r2": float(r2_score(y_test, y_pred_nn)),
            "mae": float(mean_absolute_error(y_test, y_pred_nn)),
        },
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train_proc.shape[1]),
    }

    # Guardado de artefactos
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    model.save(os.path.join("models", "model.h5"))
    joblib.dump(rf, os.path.join("models", "baseline_random_forest.joblib"))
    joblib.dump(preprocessor, os.path.join("models", "preprocessor.joblib"))

    # nombres de columnas transformadas
    with open(os.path.join("models", "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    # predictions
    pred_df = pd.DataFrame({
        "y_true": y_test.values if hasattr(y_test, "values") else y_test,
        "y_pred_baseline": y_pred_rf,
        "y_pred_nn": y_pred_nn,
    })
    pred_df.to_csv(os.path.join("results", "predictions.csv"), index=False)

    # metrics
    with open(os.path.join("results", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # history
    with open(os.path.join("results", "history_nn.json"), "w", encoding="utf-8") as f:
        json.dump(history.history, f, ensure_ascii=False, indent=2)

    # Resumen
    print("\n=== MÉTRICAS ===")
    for model_name, m in metrics.items():
        if isinstance(m, dict):
            print(f"{model_name}: R²={m['r2']:.4f} | MAE={m['mae']:.4f}")
    print(f"Registros train/test: {metrics['n_train']}/{metrics['n_test']} | Features: {metrics['n_features']}")

    return metrics


def main():
    """CLI del entrenamiento: parsea argumentos y ejecuta el pipeline de train/eval."""
    parser = argparse.ArgumentParser(description="Entrena baseline y red neuronal sobre dataset.csv")
    parser.add_argument("--data", type=str, default=os.path.join("data", "dataset.csv"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rf-estimators", type=int, default=300)
    args = parser.parse_args()

    train_and_evaluate(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        rf_estimators=args.rf_estimators,
    )


if __name__ == "__main__":
    main()
