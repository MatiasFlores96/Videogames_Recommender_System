import pandas as pd
import ast
from tqdm import tqdm

# --- 1. Función para cargar los datos ---
# Los archivos están en un formato donde cada línea es un diccionario de Python.
# Usamos ast.literal_eval para parsear cada línea de forma segura.
def cargar_datos(ruta):
    datos = []
    with open(ruta, 'r', encoding='utf-8') as f:
        for linea in tqdm(f, desc=f"Cargando {ruta}"):
            try:
                datos.append(ast.literal_eval(linea))
            except (ValueError, SyntaxError):
                # Ignorar líneas malformadas
                pass
    return datos

# Reemplaza con las rutas a tus archivos
ruta_reviews = 'Data/australian_user_reviews.json'
ruta_games = 'Data/steam_games.json'

# Cargar los datos crudos
reviews_data = cargar_datos(ruta_reviews)
games_data = cargar_datos(ruta_games)

# --- 2. Procesar y aplanar el Dataset de Reseñas ---
# La estructura es una lista de usuarios, cada uno con una lista de reseñas.
# La aplanamos para tener una fila por reseña.
reviews_aplanado = []
for user_data in tqdm(reviews_data, desc="Procesando reseñas"):
    user_id = user_data.get('user_id')
    for review in user_data.get('reviews', []):
        review_dict = {
            'user_id': user_id,
            'item_id': review.get('item_id'),
            'recommend': review.get('recommend'),
            'review_text': review.get('review')
        }
        reviews_aplanado.append(review_dict)

df_reviews = pd.DataFrame(reviews_aplanado)

# --- 3. Procesar el Dataset de Juegos ---
# Creamos el DataFrame y seleccionamos las columnas útiles.
df_games = pd.DataFrame(games_data)
df_games_clean = df_games[['id', 'app_name', 'genres', 'tags', 'developer', 'release_date', 'price']].copy()

# Renombramos la columna 'id' para que coincida con la de reseñas
df_games_clean.rename(columns={'id': 'item_id'}, inplace=True)

# --- 4. Unión de los Datasets ---
# Unimos los dos DataFrames usando 'item_id' como clave.
# Usamos un 'left' join para mantener todas las reseñas, incluso si un juego no tiene datos.
print("\nUniendo los datasets...")
df_merged = pd.merge(df_reviews, df_games_clean, on='item_id', how='left')

# --- 5. Limpieza Final y Verificación ---
# Eliminamos filas donde la reseña o el item_id son nulos, ya que son inútiles para el modelo.
df_final = df_merged.dropna(subset=['review_text', 'item_id']).copy()
# Eliminamos duplicados
df_final.drop_duplicates(subset=['user_id', 'item_id'], inplace=True)


# Convertir 'release_date' a formato de fecha
df_final['release_date'] = pd.to_datetime(df_final['release_date'], errors='coerce')

# Verificar el resultado
print("\n--- Información del DataFrame Final ---")
df_final.info()

print("\n--- Primeras 5 filas del DataFrame Final ---")
print(df_final.head())

print(f"\nProceso completado. El dataset final tiene {df_final.shape[0]} filas.")

# --- Limpieza de la columna 'price' ---
# Convertir la columna a numérico. Los errores se convertirán en NaN.
df_final['price'] = pd.to_numeric(df_final['price'], errors='coerce')

# Reemplazar los valores NaN con 0.0.
df_final['price'].fillna(0.0, inplace=True)

# --- Guardado del DataFrame ---
# Ahora esta línea debería funcionar sin problemas
df_final.to_parquet('Data/dataset_procesado.parquet')

# Guardar el DataFrame limpio y procesado para uso futuro
print("\nGuardando el DataFrame procesado en 'Data/dataset_procesado.parquet'...")
df_final.to_parquet('Data/dataset_procesado.parquet')
print("Guardado completado.")