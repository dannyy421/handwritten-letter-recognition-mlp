import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# -------------- CARGAR MODELO ENTRENADO Y TEST --------------------
try:
    # Red neuronal
    modelo_cargado = tf.keras.models.load_model("modelo_abecedario.keras")
    
    # Pruebas
    X_test = np.load("Archivos_X_y\\X_test.npy")
    y_test = np.load("Archivos_X_y\\y_test.npy")
    
    print(f"Total de imágenes disponibles en X_test: {X_test.shape[0]}")
except FileNotFoundError as e:
    print(f"No se encontro archivo: {e}")
    exit()

# Indices para visualizacion
abc = [chr(i) for i in range(ord('A'), ord('Z')+1)]

# ---------- SELECCION ALEATORIA -------------------------
total_muestras = X_test.shape[0]

# 15 al azar
indices_aleatorios = np.random.choice(total_muestras, size=15, replace=False)

# Sus imagenes correspondientes
lote_imagenes = X_test[indices_aleatorios]

# Inferencia sobre indices seleccionados
predicciones_prob = modelo_cargado.predict(lote_imagenes)
predicciones_clases = np.argmax(predicciones_prob, axis=1)

# ----------------- VISUALIZACION DE RESULTADOS --------------------
plt.figure(figsize=(12, 7))

for i, indice in enumerate(indices_aleatorios):
    plt.subplot(3, 5, i + 1)
    
    # Reconstruir la matriz de 50x50 pixeles
    imagen_2d = X_test[indice].reshape(50, 50)
    plt.imshow(imagen_2d, cmap='gray')
    
    # Decodificar etiqueta de one shot
    indice_real = np.argmax(y_test[indice]) if len(y_test.shape) > 1 else int(y_test[indice])
    letra_real = abc[indice_real]
    
    # Decodificar la predicción que hizo la red
    letra_pred = abc[predicciones_clases[i]]
    
    #Verde si acerto y rojo si fallo
    color_titulo = 'green' if letra_real == letra_pred else 'red'
    
    # Mostrar el índice original de la matriz para poder rastrearla si hay fallas
    plt.title(f"Idx en archivo: {indice}\nReal: {letra_real} | Pred: {letra_pred}", 
              color=color_titulo, fontsize=9)
    plt.axis('off')

plt.suptitle("Inferencias aleatorias sobre X_test usando modelo cargado (.keras)")
plt.tight_layout()
plt.show()