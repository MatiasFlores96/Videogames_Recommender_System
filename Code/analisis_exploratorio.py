import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- Cargar el dataset procesado ---
# Este es el único paso de carga que necesitarás de ahora en más.
print("Cargando el dataset procesado...")
df = pd.read_parquet('Data/dataset_procesado.parquet')
print("Dataset cargado exitosamente. Generando visualizaciones...")

# --- Configuración para los gráficos ---
sns.set(style="whitegrid")
plt.style.use('dark_background')

# --- 1. Distribución de Recomendaciones ---
plt.figure(figsize=(8, 5))
sns.countplot(x='recommend', data=df, palette=['#ff6347', '#3cb371'])
plt.title('Distribución de Recomendaciones (True/False)', fontsize=16)
plt.xlabel('Recomendado', fontsize=12)
plt.ylabel('Cantidad de Reseñas', fontsize=12)
plt.show()

# --- 2. Top 15 Géneros más comunes ---
# Usamos dropna() para evitar errores con juegos que no tienen géneros listados.
all_genres = Counter(g for genres_list in df['genres'].dropna() for g in genres_list)
top_genres = all_genres.most_common(15)

df_top_genres = pd.DataFrame(top_genres, columns=['Genre', 'Count'])

plt.figure(figsize=(12, 8))
sns.barplot(x='Count', y='Genre', data=df_top_genres, palette='viridis')
plt.title('Top 15 Géneros de Videojuegos', fontsize=16)
plt.xlabel('Cantidad de Juegos', fontsize=12)
plt.ylabel('Género', fontsize=12)
plt.show()

# --- 3. Top 15 Juegos con más reseñas ---
top_games = df['app_name'].value_counts().nlargest(15)

plt.figure(figsize=(12, 8))
sns.barplot(x=top_games.values, y=top_games.index, palette='rocket')
plt.title('Top 15 Juegos con Más Reseñas', fontsize=16)
plt.xlabel('Cantidad de Reseñas', fontsize=12)
plt.ylabel('Juego', fontsize=12)
plt.show()

print("\nAnálisis exploratorio visual completado.")

# --- 4. Análisis de la distribución de precios ---
# Filtramos los juegos que son gratuitos para analizar solo los de pago.
df_pagos = df[df['price'] > 0]

plt.figure(figsize=(12, 6))
sns.histplot(df_pagos['price'], bins=50, kde=True, color='skyblue')
plt.title('Distribución de Precios de Videojuegos (Precio > 0)', fontsize=16)
plt.xlabel('Precio (USD)', fontsize=12)
plt.ylabel('Cantidad de Juegos', fontsize=12)
plt.xlim(0, 100) # Limitamos el eje x para una mejor visualización
plt.show()


# --- 5. Análisis de Precios vs. Recomendaciones ---
print("\nGenerando gráfico de precios vs. recomendaciones...")

# Crear una categoría para juegos gratuitos vs. de pago
df['price_category'] = df['price'].apply(lambda x: 'Gratuito' if x == 0 else 'De Pago')

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='price_category', hue='recommend', palette=['#ff6347', '#3cb371'])
plt.title('Recomendaciones para Juegos Gratuitos vs. de Pago', fontsize=16)
plt.xlabel('Categoría de Precio', fontsize=12)
plt.ylabel('Cantidad de Reseñas', fontsize=12)
plt.legend(title='Recomendado')
plt.show()


# --- 6. Análisis del Largo de las Reseñas ---
print("\nGenerando gráfico del largo de las reseñas...")

# Calcular el largo de cada reseña en caracteres
# Usamos .str.len() que maneja correctamente los valores nulos si los hubiera
df['review_length'] = df['review_text'].str.len()

plt.figure(figsize=(12, 6))
sns.histplot(df['review_length'], bins=100, kde=True, color='purple')
plt.title('Distribución del Largo de las Reseñas (en caracteres)', fontsize=16)
plt.xlabel('Largo de la Reseña', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.xlim(0, 2000) # Limitar para mejor visualización, la mayoría de reseñas son cortas
plt.show()


# --- 7. Análisis Temporal (Lanzamientos por Año) ---
print("\nGenerando gráfico de lanzamientos por año...")

# Extraer el año de la fecha de lanzamiento
df['release_year'] = df['release_date'].dt.year

# Contar juegos por año, filtrando años con pocos datos para un gráfico más limpio
games_per_year = df['release_year'].value_counts().sort_index()
games_per_year = games_per_year[(games_per_year.index >= 2000) & (games_per_year.index <= 2018)]


plt.figure(figsize=(14, 7))
sns.lineplot(x=games_per_year.index, y=games_per_year.values, marker='o', color='gold')
plt.title('Cantidad de Juegos Lanzados por Año en el Dataset', fontsize=16)
plt.xlabel('Año de Lanzamiento', fontsize=12)
plt.ylabel('Cantidad de Juegos', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()