import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
from model import create_model


"""
    Sección: Cargar el dataset limpio (preprocesado)
    
    Qué se hace: Se carga el archivo CSV que contiene los datos preprocesados para el modelo.
    Por qué se hace: El dataset contiene las características que usaremos para entrenar y evaluar el modelo de regresión logística.
    Para qué se hace: Este dataset es necesario para poder dividir los datos en conjuntos de entrenamiento, validación y prueba.
"""
current_directory = os.path.dirname(__file__)  # Obtiene el directorio donde está train.py
data_directory = os.path.join(current_directory, 'data')  # Apunta a la carpeta 'data' donde está el CSV
file_path = os.path.join(data_directory, 'train.csv')  # Define la ruta completa del archivo CSV
df = pd.read_csv(file_path)  # Lee el dataset y lo almacena en un DataFrame


"""
    Sección: Separar características (X) y la variable objetivo (y)
    
    Qué se hace: Se separan las características (X) y la variable objetivo (y) del dataset.
    Por qué se hace: Necesitamos identificar qué columnas del dataset representan las características (entradas) y cuál es la variable a predecir.
    Para qué se hace: Para entrenar el modelo, necesitamos un conjunto de datos que contenga solo las características y otro con las etiquetas (target).
"""
X = df.drop(columns=['target'])  # 'X' contiene todas las columnas menos 'target'
y = df['target']  # 'y' es la columna que contiene la variable objetivo (target)


"""
    Sección: Dividir el dataset en train (70%), validation (15%) y test (15%)
    
    Qué se hace: Se divide el dataset en tres conjuntos: entrenamiento, validación y prueba.
    Por qué se hace: Para evaluar el rendimiento del modelo de manera adecuada, es necesario entrenarlo en un conjunto y evaluarlo en otros datos no vistos.
    Para qué se hace: Esta división permite evaluar cómo generaliza el modelo a datos que no ha visto, ayudando a evitar el sobreajuste.
"""
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # Dividimos en train (70%) y un conjunto temporal (30%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Dividimos el temporal en validación y prueba (15% cada uno)


"""
    Sección: Escalar los datos
    
    Qué se hace: Se escalan los datos usando StandardScaler.
    Por qué se hace: Muchos algoritmos de machine learning, incluida la regresión logística, funcionan mejor cuando los datos están en la misma escala.
    Para qué se hace: El escalado mejora el rendimiento y la estabilidad del modelo, ya que características en escalas muy diferentes pueden afectar negativamente los coeficientes del modelo.
"""
scaler = StandardScaler()  # Inicializamos el objeto de escalado
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)  # Escalamos los datos de entrenamiento y mantenemos los nombres de columnas
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)  # Escalamos los datos de validación
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)  # Escalamos los datos de prueba


"""
    Sección: Crear y entrenar el modelo
    
    Qué se hace: Se crea y entrena el modelo de regresión logística utilizando los datos escalados.
    Por qué se hace: El modelo de regresión logística es adecuado para problemas de clasificación binaria, como predecir si una canción es buena o mala.
    Para qué se hace: El entrenamiento permite al modelo aprender patrones a partir de los datos de entrenamiento para hacer predicciones futuras.
"""
model = create_model()  # Se crea el modelo utilizando los parámetros definidos en la función create_model
model.fit(X_train_scaled, y_train)  # Entrenamos el modelo con los datos de entrenamiento


"""
    Sección: Validación cruzada
    
    Qué se hace: Se realiza una validación cruzada de 5 particiones sobre los datos de entrenamiento.
    Por qué se hace: La validación cruzada ayuda a evaluar la estabilidad del modelo y asegura que no se sobreajuste a un conjunto particular de datos.
    Para qué se hace: Proporciona una estimación más robusta del rendimiento del modelo, reduciendo la dependencia de un solo conjunto de datos.
"""
cross_val_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)  # Realizamos 5-fold cross-validation
print(f'Cross-validation accuracy: {cross_val_scores.mean():.2f}%')  # Imprimimos la exactitud promedio de la validación cruzada


"""
    Sección: Evaluar el modelo en el conjunto de validación
    
    Qué se hace: Se evalúa el modelo utilizando los datos de validación.
    Por qué se hace: Para verificar el rendimiento del modelo en datos que no ha visto durante el entrenamiento.
    Para qué se hace: Esto permite ajustar el modelo antes de evaluar su rendimiento final en el conjunto de prueba.
"""
y_val_pred = model.predict(X_val_scaled)  # Realizamos predicciones en el conjunto de validación


# Métricas de validación: Exactitud, Precisión, Recall, F1

# Exactitud (Accuracy):
# Qué es: Es la proporción de predicciones correctas sobre el total de predicciones realizadas.
# Por qué se utiliza: Es una métrica simple que indica qué tan bien el modelo clasifica en general, 
# pero puede ser engañosa si las clases están desbalanceadas, ya que favorece a la clase mayoritaria.
# Para qué nos sirve: Nos da una visión general de cuántas predicciones fueron correctas, 
# pero no es ideal cuando una clase domina sobre otra.
accuracy_val = accuracy_score(y_val, y_val_pred)

# Precisión (Precision):
# Qué es: Es la proporción de verdaderos positivos sobre todos los ejemplos que fueron clasificados como positivos 
# (es decir, la precisión en las predicciones de la clase positiva).
# Por qué se utiliza: Es útil cuando queremos minimizar los falsos positivos. 
# En este caso, nos ayuda a medir cuántas de las canciones clasificadas como "buenas" realmente lo son.
# Para qué nos sirve: Si nos importa más ser correctos al predecir una canción como buena, esta métrica nos ayuda a entenderlo mejor.
precision_val = precision_score(y_val, y_val_pred)

# Recall (Sensibilidad o Tasa de Verdaderos Positivos):
# Qué es: Es la proporción de verdaderos positivos sobre todos los ejemplos que son realmente positivos.
# Por qué se utiliza: Es importante cuando queremos minimizar los falsos negativos. 
# En este caso, nos dice cuántas de las canciones que son "buenas" fueron correctamente identificadas como tales.
# Para qué nos sirve: Nos indica si el modelo está perdiendo canciones buenas al clasificarlas incorrectamente como malas.
recall_val = recall_score(y_val, y_val_pred)

# F1 Score:
# Qué es: Es la media armónica entre la precisión y el recall, y es útil cuando buscamos un balance entre ambos.
# Por qué se utiliza: Cuando hay un desbalance entre precisión y recall, F1 score es una buena métrica que equilibra ambos.
# Para qué nos sirve: Nos da una visión general del rendimiento del modelo cuando buscamos un equilibrio entre evitar falsos positivos y falsos negativos.
f1_val = f1_score(y_val, y_val_pred)

# Imprimimos los resultados de validación
print(f'Accuracy en validación: {accuracy_val * 100:.2f}%')
print(f'Precision en validación: {precision_val * 100:.2f}%')
print(f'Recall en validación: {recall_val * 100:.2f}%')
print(f'F1 Score en validación: {f1_val * 100:.2f}%')


# Evaluar en el conjunto de prueba
# Qué se hace: Se realizan predicciones en el conjunto de prueba utilizando el modelo entrenado.
# Por qué se hace: El conjunto de prueba contiene datos que el modelo no ha visto antes. Evaluar el rendimiento en este conjunto nos permite medir la capacidad de generalización del modelo.
# Para qué se hace: Para verificar si el modelo puede hacer predicciones precisas en datos nuevos y diferentes de los utilizados en el entrenamiento o validación.
y_test_pred = model.predict(X_test_scaled)

# Métricas en el conjunto de prueba

# Exactitud (Accuracy):
# Qué es: Es la proporción de predicciones correctas en relación al total de predicciones realizadas.
# Por qué se utiliza: Nos da una visión general de qué tan bien predice el modelo en el conjunto de prueba. Sin embargo, como en la validación, puede ser menos confiable si las clases están desbalanceadas.
# Para qué nos sirve: Sirve para entender el rendimiento global del modelo, aunque no revela problemas con clases desbalanceadas.
accuracy_test = accuracy_score(y_test, y_test_pred)

# Precisión (Precision):
# Qué es: Es la proporción de verdaderos positivos sobre todas las predicciones de la clase positiva.
# Por qué se utiliza: Es útil si es más importante evitar falsos positivos, por ejemplo, si clasificar incorrectamente una canción como "buena" puede tener un impacto negativo.
# Para qué nos sirve: Nos muestra cuántas de las predicciones que el modelo hizo como "buenas" son efectivamente correctas.
precision_test = precision_score(y_test, y_test_pred)

# Recall (Sensibilidad o Tasa de Verdaderos Positivos):
# Qué es: Es la proporción de verdaderos positivos sobre todos los ejemplos que son realmente positivos.
# Por qué se utiliza: Es importante cuando queremos evitar falsos negativos, asegurándonos de que la mayoría de las canciones "buenas" sean identificadas correctamente.
# Para qué nos sirve: Nos dice si el modelo está clasificando correctamente las canciones "buenas" o está omitiendo algunas que realmente lo son.
recall_test = recall_score(y_test, y_test_pred)

# F1 Score:
# Qué es: Es la media armónica entre la precisión y el recall. Es una métrica balanceada que combina ambas y es útil cuando se busca un equilibrio entre evitar falsos positivos y falsos negativos.
# Por qué se utiliza: En muchos casos, ni la precisión ni el recall por sí solos son suficientes, y el F1 Score ayuda a encontrar un equilibrio entre ambos.
# Para qué nos sirve: Nos da una visión más completa del rendimiento del modelo cuando es importante tanto predecir correctamente las canciones buenas como evitar falsos positivos.
f1_test = f1_score(y_test, y_test_pred)

# Imprimir resultados del conjunto de prueba
# Qué se hace: Se imprimen los resultados de las métricas calculadas.
# Por qué se hace: Para visualizar el rendimiento del modelo en datos no vistos (conjunto de prueba) y compararlo con el rendimiento en el conjunto de validación.
# Para qué se hace: Para entender cómo de bien generaliza el modelo a datos nuevos y determinar si es adecuado para producción.
print(f'\n--- Evaluación en el conjunto de prueba ---')
print(f'Accuracy en prueba: {accuracy_test * 100:.2f}%')
print(f'Precision en prueba: {precision_test * 100:.2f}%')
print(f'Recall en prueba: {recall_test * 100:.2f}%')
print(f'F1 Score en prueba: {f1_test * 100:.2f}%')



"""
    Sección: Matriz de confusión
    
    Qué se hace: Se calcula e imprime la matriz de confusión.
    Por qué se hace: La matriz de confusión muestra cuántos ejemplos fueron clasificados correctamente e incorrectamente en cada clase.
    Para qué se hace: Proporciona una visión más detallada del rendimiento del modelo en términos de verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos.
"""
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("\nMatriz de confusión:")
print(conf_matrix)


"""
    Sección: Guardar el modelo entrenado
    
    Qué se hace: Se guarda el modelo entrenado en un archivo .pkl.
    Por qué se hace: Guardar el modelo permite reutilizarlo más adelante sin necesidad de volver a entrenarlo.
    Para qué se hace: Para ahorrar tiempo y poder utilizar el modelo entrenado directamente para hacer predicciones en nuevos datos.
"""
model_directory = os.path.join(current_directory, 'model')  # Apunta a la carpeta 'model'
model_path = os.path.join(model_directory, 'modeloRegresionFramework.pkl')  # Define el nombre del archivo .pkl

# Guardar el modelo entrenado en formato pickle
with open(model_path, 'wb') as f:
    pickle.dump(model, f)  # Guarda el modelo entrenado en el archivo

