from pathlib import Path
import sys

# Permite importar 'eda' sin importar desde qué directorio se ejecute el script
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
  sys.path.append(str(SRC_DIR))

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC

from eda import cargar_datos, limpiar_datos

MODELS_DIR = BASE_DIR / 'models'

COLUMNAS_X = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol',
]
COLUMNA_Y = 'quality'


def dividir_datos(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 0):
  """Separa variables predictoras y objetivo aplicando división estratificada."""
  X = df[COLUMNAS_X].values
  y = df[COLUMNA_Y].values

  return train_test_split(
      X, y, test_size=test_size, random_state=random_state, stratify=y
  )


def construir_pipelines() -> dict[str, Pipeline]:
  """Define los 5 modelos encapsulados con su transformación Yeo-Johnson + escalado."""
  clasificadores = {
      'Random Forest': RandomForestClassifier(
          n_estimators=100,
          criterion='entropy',
          random_state=42,
          class_weight='balanced',
      ),
      'Regresión Logística': LogisticRegression(
          max_iter=1000, class_weight='balanced', random_state=42
      ),
      'Support Vector Machine (SVC)': SVC(
          kernel='rbf', C=1.0, class_weight='balanced', random_state=42
      ),
      'K-Nearest Neighbors (KNN)': KNeighborsClassifier(
          n_neighbors=7, weights='distance', metric='minkowski'
      ),
      'Gradient Boosting': GradientBoostingClassifier(
          n_estimators=100, learning_rate=0.1, random_state=42
      ),
  }

  pipelines = {}
  for nombre, clf in clasificadores.items():
    pipelines[nombre] = Pipeline([
        ('transformador', PowerTransformer(method='yeo-johnson', standardize=True)),
        ('clasificador', clf),
    ])
  return pipelines


def entrenar_y_evaluar(X_train, X_test, y_train, y_test):
  """Entrena todos los pipelines, calcula métricas y selecciona el mejor modelo."""
  pipelines = construir_pipelines()
  resultados = []
  modelos_entrenados = {}

  for nombre, pipeline in pipelines.items():
    print('=' * 60)
    print(f'ENTRENANDO MODELO: {nombre}')
    print('=' * 60)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)

    resultados.append({
        'Modelo': nombre,
        'Accuracy': round(acc, 4),
        'Precision (Weighted)': round(prec, 4),
        'Recall (Weighted)': round(rec, 4),
        'F1-Score (Weighted)': round(f1_w, 4),
        'F1-Score (Macro)': round(f1_m, 4),
    })
    modelos_entrenados[nombre] = pipeline

    print(classification_report(y_test, y_pred, zero_division=0))

  df_resultados = pd.DataFrame(resultados).sort_values(
      by='F1-Score (Weighted)', ascending=False
  )

  print('\n' + '=' * 60)
  print('RESUMEN COMPARATIVO DE MODELOS')
  print('=' * 60)
  print(df_resultados.to_string(index=False))

  mejor_nombre = df_resultados.iloc[0]['Modelo']
  mejor_modelo = modelos_entrenados[mejor_nombre]
  return mejor_nombre, mejor_modelo


def guardar_artefactos(
    nombre_modelo: str, modelo: Pipeline, X_test, y_test
) -> None:
  """Guarda en disco el mejor pipeline entrenado y los datos de prueba."""
  MODELS_DIR.mkdir(parents=True, exist_ok=True)

  ruta_modelo = MODELS_DIR / 'mejor_modelo.joblib'
  ruta_datos_prueba = MODELS_DIR / 'datos_prueba.joblib'

  joblib.dump({'nombre': nombre_modelo, 'pipeline': modelo}, ruta_modelo)
  joblib.dump({'X_test': X_test, 'y_test': y_test}, ruta_datos_prueba)

  print(f"\nMejor modelo ('{nombre_modelo}') guardado en: {ruta_modelo}")
  print(f'Conjunto de prueba guardado en: {ruta_datos_prueba}')


def ejecutar_entrenamiento() -> None:
  """Flujo completo de entrenamiento y selección de modelo."""
  df = cargar_datos()
  df_limpio = limpiar_datos(df)
  X_train, X_test, y_train, y_test = dividir_datos(df_limpio)
  mejor_nombre, mejor_modelo = entrenar_y_evaluar(
      X_train, X_test, y_train, y_test
  )
  guardar_artefactos(mejor_nombre, mejor_modelo, X_test, y_test)


if __name__ == '__main__':
  ejecutar_entrenamiento()