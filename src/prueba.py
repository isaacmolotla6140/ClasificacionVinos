from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'


def cargar_artefactos():
  """Carga el modelo guardado y el conjunto de prueba desde disco."""
  ruta_modelo = MODELS_DIR / 'mejor_modelo.joblib'
  ruta_datos_prueba = MODELS_DIR / 'datos_prueba.joblib'

  if not ruta_modelo.exists() or not ruta_datos_prueba.exists():
    raise FileNotFoundError(
        'No se encontraron los archivos del modelo. Ejecuta primero'
        ' src/entrenamiento.py'
    )

  artefacto_modelo = joblib.load(ruta_modelo)
  datos_prueba = joblib.load(ruta_datos_prueba)

  return (
      artefacto_modelo['nombre'],
      artefacto_modelo['pipeline'],
      datos_prueba['X_test'],
      datos_prueba['y_test'],
  )


def graficar_matriz_confusion(
    cm: np.ndarray, clases: np.ndarray, nombre_modelo: str
) -> None:
  """Muestra gráficamente la matriz de confusión."""
  plt.figure(figsize=(8, 6))
  sns.heatmap(
      cm,
      annot=True,
      fmt='d',
      cmap='Blues',
      xticklabels=clases,
      yticklabels=clases,
  )
  plt.title(f'Matriz de Confusión - {nombre_modelo}')
  plt.xlabel('Predicción')
  plt.ylabel('Valor Real')
  plt.tight_layout()
  plt.show()


def evaluar_modelo() -> None:
  """Evalúa el modelo guardado en los datos de prueba y reporta resultados."""
  nombre_modelo, pipeline, X_test, y_test = cargar_artefactos()

  y_pred = pipeline.predict(X_test)
  clases = np.sort(np.unique(np.concatenate([y_test, y_pred])))
  cm = confusion_matrix(y_test, y_pred, labels=clases)

  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
  rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
  f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
  f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)

  print('=' * 60)
  print(f'EVALUACIÓN FINAL EN PRUEBA: {nombre_modelo}')
  print('=' * 60)
  print(f'Accuracy (Exactitud):      {acc:.4f}')
  print(f'Precision (Weighted):      {prec:.4f}')
  print(f'Recall (Weighted):         {rec:.4f}')
  print(f'F1-Score (Weighted):       {f1_w:.4f}')
  print(f'F1-Score (Macro):          {f1_m:.4f}')

  print('\nMatriz de Confusión:')
  print(cm)

  print('\nReporte de Clasificación por Clase:')
  print(classification_report(y_test, y_pred, zero_division=0))

  graficar_matriz_confusion(cm, clases, nombre_modelo)


if __name__ == '__main__':
  evaluar_modelo()