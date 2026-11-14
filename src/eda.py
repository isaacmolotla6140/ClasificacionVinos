from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent


def obtener_ruta_dataset() -> Path:
  """Localiza WineQT.csv en la raíz del proyecto o dentro de una carpeta data/."""
  rutas_posibles = [
      BASE_DIR / 'WineQT.csv',
      BASE_DIR / 'data' / 'WineQT.csv',
  ]
  for ruta in rutas_posibles:
    if ruta.exists():
      return ruta
  raise FileNotFoundError(
      'No se encontró WineQT.csv en la raíz ni en data/WineQT.csv'
  )


def cargar_datos(ruta: Path | None = None) -> pd.DataFrame:
  """Carga el dataset desde disco."""
  ruta_csv = ruta or obtener_ruta_dataset()
  df = pd.read_csv(ruta_csv)
  print(
      f'Dataset cargado correctamente: {df.shape[0]} filas y'
      f' {df.shape[1]} columnas.'
  )
  return df


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
  """Elimina registros duplicados y columnas irrelevantes como 'Id'."""
  df_limpio = df.copy()

  duplicados = df_limpio.duplicated().sum()
  df_limpio = df_limpio.drop_duplicates()
  print(f'Duplicados eliminados: {duplicados}')

  if 'Id' in df_limpio.columns:
    df_limpio = df_limpio.drop(columns=['Id'])
    print("Columna 'Id' eliminada.")

  return df_limpio


def mostrar_estadisticas(df: pd.DataFrame) -> None:
  """Imprime información general, valores nulos, conteo de clases y estadísticos."""
  print('\n--- Primeras 5 filas ---')
  print(df.head())

  print('\n--- Información general ---')
  df.info()

  print('\n--- Valores nulos por columna ---')
  print(df.isnull().sum())

  print('\n--- Estadísticas descriptivas ---')
  print(df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']])

  if 'quality' in df.columns:
    print('\n--- Conteo por clase en variable objetivo (quality) ---')
    print(df['quality'].value_counts().sort_index())


def visualizar_distribuciones(df: pd.DataFrame) -> None:
  """Genera histogramas, conteo de clases, boxplots y matriz de correlación."""
  variables_predictoras = df.drop(columns=['quality']).columns

  # A) Histogramas de las variables predictoras
  df[variables_predictoras].hist(
      bins=20, figsize=(16, 12), color='#4C72B0', edgecolor='black'
  )
  plt.suptitle('Distribución de variables fisicoquímicas (Histogramas)', y=1.02)
  plt.tight_layout()
  plt.show()

  # B) Distribución de la variable objetivo (quality)
  plt.figure(figsize=(8, 4))
  sns.countplot(
      data=df, x='quality', hue='quality', palette='viridis', legend=False
  )
  plt.title('Frecuencia de calificaciones de calidad del vino')
  plt.xlabel('Calidad (quality)')
  plt.ylabel('Cantidad de muestras')
  plt.tight_layout()
  plt.show()

  # C) Boxplots para identificar valores atípicos (outliers)
  plt.figure(figsize=(16, 10))
  for i, col in enumerate(variables_predictoras, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(y=df[col], color='#DD8452')
    plt.title(col)
    plt.ylabel('')
  plt.suptitle('Detección de valores atípicos (Boxplots)', y=1.02)
  plt.tight_layout()
  plt.show()

  # D) Matriz de correlación
  plt.figure(figsize=(12, 9))
  matriz_corr = df.corr()
  sns.heatmap(
      matriz_corr,
      annot=True,
      fmt='.2f',
      cmap='coolwarm',
      linewidths=0.5,
      vmin=-1,
      vmax=1,
  )
  plt.title('Matriz de Correlación entre Variables')
  plt.tight_layout()
  plt.show()


def ejecutar_eda() -> pd.DataFrame:
  """Flujo completo del Análisis Exploratorio de Datos."""
  df = cargar_datos()
  mostrar_estadisticas(df)
  df_limpio = limpiar_datos(df)
  visualizar_distribuciones(df_limpio)
  return df_limpio


if __name__ == '__main__':
  ejecutar_eda()