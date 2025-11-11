import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")


def ensure_dirs():
    """Crea el directorio de resultados si no existe."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_scatter(df: pd.DataFrame, col_pred: str, out_path: Path, title: str):
    """Genera un diagrama de dispersión y=y_pred vs x=y_true con línea identidad.
    Params:
        df: DataFrame con columnas 'y_true' y la predicción indicada por col_pred.
        col_pred: nombre de la columna de predicción a graficar.
        out_path: ruta donde guardar la imagen PNG.
        title: título de la figura.
    """
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=df["y_true"], y=df[col_pred], s=20, alpha=0.6)
    lims = [min(df["y_true"].min(), df[col_pred].min()), max(df["y_true"].max(), df[col_pred].max())]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("y_true (precio)")
    plt.ylabel(col_pred)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_residuals(df: pd.DataFrame, col_pred: str, out_path: Path, title: str):
    """Grafica histograma de residuales (pred - true) con KDE.
    Params:
        df: DataFrame con 'y_true' y la columna de predicción.
        col_pred: nombre de la columna de predicción.
        out_path: ruta de salida del PNG.
        title: título de la figura.
    """
    residuals = df[col_pred] - df["y_true"]
    plt.figure(figsize=(7, 4))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residual (pred - true)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importances(out_path: Path, top_k: int = 20):
    """Dibuja un barplot con las top-k importancias del RandomForest entrenado.
    Si no existen el modelo baseline o los nombres de features, no hace nada.
    """
    rf_path = MODELS_DIR / "baseline_random_forest.joblib"
    feat_path = MODELS_DIR / "feature_names.json"
    if not rf_path.exists() or not feat_path.exists():
        return
    rf = joblib.load(rf_path)
    with open(feat_path, "r", encoding="utf-8") as f:
        feat_names = json.load(f)
    importances = getattr(rf, "feature_importances_", None)
    if importances is None:
        return
    imp = pd.DataFrame({"feature": feat_names, "importance": importances})
    imp = imp.sort_values("importance", ascending=False).head(top_k)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=imp, y="feature", x="importance")
    plt.title("Importancias (RandomForest) - Top 20")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_training_history(history_path: Path, out_path: Path):
    """Grafica la historia de entrenamiento de la NN (loss y val_loss por epoch)."""
    if not history_path.exists():
        return
    with open(history_path, "r", encoding="utf-8") as f:
        hist = json.load(f)
    loss = hist.get("loss", [])
    val_loss = hist.get("val_loss", [])
    plt.figure(figsize=(7, 4))
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.title("Historia de entrenamiento (NN)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    """CLI de evaluación: carga predicciones/métricas, genera gráficas y muestra resumen."""
    ensure_dirs()
    pred_path = RESULTS_DIR / "predictions.csv"
    metrics_path = RESULTS_DIR / "metrics.json"

    if not pred_path.exists():
        print("No se encontró results/predictions.csv. Ejecuta primero train_model.py.")
        return

    df = pd.read_csv(pred_path)

    # Gráficas de dispersión y residuales
    plot_scatter(df, "y_pred_baseline", RESULTS_DIR / "scatter_baseline.png", "y_true vs y_pred (Baseline)")
    plot_scatter(df, "y_pred_nn", RESULTS_DIR / "scatter_nn.png", "y_true vs y_pred (NN)")
    plot_residuals(df, "y_pred_baseline", RESULTS_DIR / "residuals_baseline.png", "Residuales (Baseline)")
    plot_residuals(df, "y_pred_nn", RESULTS_DIR / "residuals_nn.png", "Residuales (NN)")

    # Importancias
    plot_feature_importances(RESULTS_DIR / "feature_importances_top20.png", top_k=20)

    # Historia de entrenamiento
    plot_training_history(RESULTS_DIR / "history_nn.json", RESULTS_DIR / "training_history.png")

    # Mostrar métricas en consola
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        print("\n=== MÉTRICAS ===")
        for model_name, m in metrics.items():
            if isinstance(m, dict) and "r2" in m:
                print(f"{model_name}: R²={m['r2']:.4f} | MAE={m['mae']:.4f}")


if __name__ == "__main__":
    main()
