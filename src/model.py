# Importamos la Regresión Logística
from sklearn.linear_model import LogisticRegression

"""
    Esta función crea y devuelve un modelo de regresión logística ajustado para resolver un problema de clasificación binaria.
    
    Se utiliza el solver 'saga', que es eficiente para grandes datasets y admite regularización tanto L1 como L2.
    Además, se aplica una regularización fuerte (C=0.01) para evitar el sobreajuste y mejorar la capacidad de generalización del modelo.

    Se establece un número de iteraciones elevado (500) para garantizar que el solver tenga suficiente tiempo para converger.
    El modelo utiliza la regularización L2 (Ridge) para penalizar los coeficientes grandes y mejorar la estabilidad.

    Se emplea 'class_weight=balanced' para ajustar los pesos automáticamente en función de la frecuencia de las clases, 
    lo cual es útil cuando las clases están desbalanceadas (por ejemplo, si hay más canciones malas que buenas).
    
    Finalmente, la semilla 'random_state=42' se establece para asegurar que los resultados sean reproducibles.
"""
def create_model():
    model = LogisticRegression(solver='saga', C=0.01, random_state=42, max_iter=500, penalty="l2", class_weight='balanced')
    return model