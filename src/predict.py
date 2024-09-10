import pandas as pd
import pickle
import numpy as np
import os

"""
    Sección: Cargar el modelo entrenado
    
    Qué se hace: Se carga el modelo previamente entrenado desde un archivo `.pkl`.
    Por qué se hace: Para reutilizar el modelo sin tener que entrenarlo de nuevo.
    Para qué se hace: Para hacer predicciones sobre nuevos datos, sin tener que volver a ejecutar el entrenamiento.
"""
current_directory = os.path.dirname(__file__)  # Obtiene el directorio donde está predict.py
model_directory = os.path.join(current_directory, 'model')  # Apunta a la carpeta 'model'
model_path = os.path.join(model_directory, 'modeloRegresionFramework.pkl')  # Ruta completa al archivo del modelo

with open(model_path, 'rb') as f:  # Abre el archivo del modelo en modo lectura binaria
    model = pickle.load(f)  # Carga el modelo desde el archivo pickle



"""
    Sección: Generar datos aleatorios para probar
    
    Qué se hace: Se generan ejemplos aleatorios de datos de canciones, con características similares a las utilizadas para entrenar el modelo.
    Por qué se hace: Para simular nuevos datos y verificar cómo el modelo hace predicciones con datos no vistos.
    Para qué se hace: Esto permite probar el modelo con ejemplos nuevos y ver si hace predicciones correctas.
"""
num_rows = 1  # Número de ejemplos a generar
test_data = {
    'popularity': np.random.randint(0, 100, size=num_rows),  # Popularidad aleatoria entre 0 y 100
    'duration_ms': np.random.randint(120000, 300000, size=num_rows),  # Duración entre 2 y 5 minutos
    'explicit': np.random.choice([0, 1], size=num_rows),  # Canción explícita (0 = no, 1 = sí)
    'danceability': np.random.uniform(0, 1, size=num_rows),  # Bailabilidad entre 0 y 1
    'energy': np.random.uniform(0, 1, size=num_rows),  # Energía entre 0 y 1
    'key': np.random.randint(0, 12, size=num_rows),  # Tonalidad (0-11)
    'loudness': np.random.uniform(-60, 0, size=num_rows),  # Volumen en dB (entre -60 y 0)
    'mode': np.random.choice([0, 1], size=num_rows),  # Modo (0 = menor, 1 = mayor)
    'speechiness': np.random.uniform(0, 1, size=num_rows),  # Cantidad de palabras en la canción
    'acousticness': np.random.uniform(0, 1, size=num_rows),  # Nivel de acústica
    'instrumentalness': np.random.uniform(0, 1, size=num_rows),  # Nivel de instrumentalidad
    'liveness': np.random.uniform(0, 1, size=num_rows),  # Presencia en vivo
    'valence': np.random.uniform(0, 1, size=num_rows),  # Emoción positiva
    'tempo': np.random.uniform(60, 200, size=num_rows),  # Tempo (beats por minuto)
    'time_signature': np.random.choice([3, 4, 5], size=num_rows),  # Compás de la canción
}


"""
    Sección: Agregar géneros musicales a los datos
    
    Qué se hace: Se agregan columnas correspondientes a diferentes géneros musicales y se selecciona uno aleatoriamente para cada ejemplo.
    Por qué se hace: Los géneros son una característica importante del dataset, y cada canción pertenece a un género.
    Para qué se hace: Para simular una canción perteneciente a un género particular, como se espera en el modelo entrenado.
"""
genres = [
    'track_genre_afrobeat', 'track_genre_alt-rock', 'track_genre_alternative', 'track_genre_ambient', 'track_genre_anime',
    'track_genre_black-metal', 'track_genre_bluegrass', 'track_genre_blues', 'track_genre_brazil', 'track_genre_breakbeat',
    'track_genre_british', 'track_genre_cantopop', 'track_genre_chicago-house', 'track_genre_children', 'track_genre_chill',
    'track_genre_classical', 'track_genre_club', 'track_genre_comedy', 'track_genre_country', 'track_genre_dance',
    'track_genre_dancehall', 'track_genre_death-metal', 'track_genre_deep-house', 'track_genre_detroit-techno', 'track_genre_disco',
    'track_genre_disney', 'track_genre_drum-and-bass', 'track_genre_dub', 'track_genre_dubstep', 'track_genre_edm',
    'track_genre_electro', 'track_genre_electronic', 'track_genre_emo', 'track_genre_folk', 'track_genre_forro',
    'track_genre_french', 'track_genre_funk', 'track_genre_garage', 'track_genre_german', 'track_genre_gospel',
    'track_genre_goth', 'track_genre_grindcore', 'track_genre_groove', 'track_genre_grunge', 'track_genre_guitar',
    'track_genre_happy', 'track_genre_hard-rock', 'track_genre_hardcore', 'track_genre_hardstyle', 'track_genre_heavy-metal',
    'track_genre_hip-hop', 'track_genre_honky-tonk', 'track_genre_house', 'track_genre_idm', 'track_genre_indian',
    'track_genre_indie', 'track_genre_indie-pop', 'track_genre_industrial', 'track_genre_iranian', 'track_genre_j-dance',
    'track_genre_j-idol', 'track_genre_j-pop', 'track_genre_j-rock', 'track_genre_jazz', 'track_genre_k-pop',
    'track_genre_kids', 'track_genre_latin', 'track_genre_latino', 'track_genre_malay', 'track_genre_mandopop',
    'track_genre_metal', 'track_genre_metalcore', 'track_genre_minimal-techno', 'track_genre_mpb', 'track_genre_new-age',
    'track_genre_opera', 'track_genre_pagode', 'track_genre_party', 'track_genre_piano', 'track_genre_pop',
    'track_genre_pop-film', 'track_genre_power-pop', 'track_genre_progressive-house', 'track_genre_psych-rock', 'track_genre_punk',
    'track_genre_punk-rock', 'track_genre_r-n-b', 'track_genre_reggae', 'track_genre_reggaeton', 'track_genre_rock',
    'track_genre_rock-n-roll', 'track_genre_rockabilly', 'track_genre_romance', 'track_genre_sad', 'track_genre_salsa',
    'track_genre_samba', 'track_genre_sertanejo', 'track_genre_show-tunes', 'track_genre_singer-songwriter', 'track_genre_ska',
    'track_genre_sleep', 'track_genre_songwriter', 'track_genre_soul', 'track_genre_spanish', 'track_genre_study',
    'track_genre_swedish', 'track_genre_synth-pop', 'track_genre_tango', 'track_genre_techno', 'track_genre_trance',
    'track_genre_trip-hop', 'track_genre_turkish', 'track_genre_world-music'
]


# Inicializar todas las columnas de géneros a 0
for genre in genres:
    test_data[genre] = np.zeros(num_rows)

# Para cada fila, seleccionar un género aleatorio y poner un 1 en esa columna
for i in range(num_rows):
    random_genre = np.random.choice(genres)  # Seleccionamos un género aleatorio
    test_data[random_genre][i] = 1  # Asignamos el género a la canción


"""
    Sección: Crear DataFrame y realizar predicciones
    
    Qué se hace: Se crea un DataFrame con los datos generados y se utilizan para realizar predicciones con el modelo cargado.
    Por qué se hace: El DataFrame es el formato que el modelo espera para realizar predicciones.
    Para qué se hace: Para evaluar si el modelo predice correctamente si la canción es "buena" o "mala".
"""
test_df = pd.DataFrame(test_data)  # Convertimos los datos generados en un DataFrame

# Realizar predicciones
predictions = model.predict(test_df)  # Usamos el modelo para predecir si las canciones son buenas o malas

# Agregar las predicciones al DataFrame
test_df['prediction'] = predictions  # Añadimos la predicción al DataFrame

# Definir qué es una "buena" canción (si la predicción es 1, es buena)
# Qué se hace: Se agrega una columna adicional en el DataFrame que clasifica la canción como "Buena canción" o "Mala canción" según la predicción.
# Por qué se hace: Queremos saber si el modelo predice correctamente si la canción es buena o mala.
# Para qué se hace: Esto nos permite interpretar más fácilmente los resultados de las predicciones en un formato legible.
test_df['is_good_song'] = test_df['prediction'].apply(lambda x: 'Buena canción' if x == 1 else 'Mala canción')

# Imprimir las características de cada canción junto con la predicción
# Qué se hace: Se imprime cada fila del DataFrame con las características de la canción y la predicción realizada por el modelo.
# Por qué se hace: Para verificar manualmente qué valores se generaron y cómo el modelo los clasificó.
# Para qué se hace: Esto permite una revisión rápida de los resultados y el comportamiento del modelo en datos aleatorios.
for index, row in test_df.iterrows():
    print(f"Canción {index + 1}:")
    print(row)
    print(f"Predicción: {row['is_good_song']}")
    print("-" * 50)


"""
    Sección: Probar con un conjunto de datos específico (buena canción)
    
    Qué se hace: Se crea un conjunto de datos específico que representa una "buena canción" y se pasa al modelo para verificar si lo predice correctamente.
    Por qué se hace: Queremos probar el comportamiento del modelo en un caso controlado, donde se sabe que la canción tiene características que deberían ser de una buena canción.
    Para qué se hace: Esto nos ayuda a ver si el modelo está bien ajustado y si es capaz de predecir correctamente casos que son intuitivamente claros.
"""
test_data_good_song = [95, 0.85, 1, 0.85, 0.9, 5, 0.7, 1, 0.05, 0.2, 0.1, 0.8, 0.9, 0.7, 4, 
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


"""
    Sección: Asegurar que el conjunto de datos tiene el número correcto de características

    Qué se hace: Se asegura que el conjunto de datos `test_data_good_song` tiene el mismo número de características que el modelo espera (128 características).
    Por qué se hace: El modelo espera un número fijo de características para hacer predicciones, y si faltan columnas, no podrá procesar los datos.
    Para qué se hace: Para asegurarse de que el conjunto de datos esté correctamente preparado antes de hacer la predicción.
"""
test_data_full = np.zeros(128)  # Inicializar array de 128 características con ceros
test_data_full[:len(test_data_good_song)] = test_data_good_song  # Insertar los valores conocidos de la buena canción


"""
    Sección: Realizar la predicción con los datos de la buena canción

    Qué se hace: Se convierte `test_data_full` a un array de NumPy y se pasa al modelo para hacer una predicción.
    Por qué se hace: Queremos verificar si el modelo clasifica correctamente la canción con características de una buena canción.
    Para qué se hace: Esto permite probar si el modelo está generalizando bien o si está sesgado hacia la clasificación de ciertas canciones.
"""
test_data_array = test_data_full.reshape(1, -1)  # Redimensionamos el array para pasarlo al modelo

# Hacer la predicción con el modelo ya entrenado
prediction = model.predict(test_data_array)  # Realizamos la predicción sobre la buena canción

# Mostrar la predicción
# Qué se hace: Se imprime la predicción del modelo (1 para "Buena canción" y 0 para "Mala canción").
# Por qué se hace: Para verificar si el modelo predice correctamente los datos de la canción generada.
# Para qué se hace: Esto sirve como prueba de que el modelo puede clasificar correctamente ejemplos controlados.
print(f'Predicción 2: {prediction[0]}')  # Esto debería dar 1 si el modelo lo predice correctamente.