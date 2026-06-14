import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


# Cargar achivos de prueba y archivos de entrenamiento
try:
    X_train = np.load("Archivos_X_y\\X_train.npy")
    y_train = np.load("Archivos_X_y\\y_train.npy")
    X_test = np.load("Archivos_X_y\\X_test.npy")
    y_test = np.load("Archivos_X_y\\y_test.npy")
    print(f"Conjunto de entrenamiento: {X_train.shape}")
    print(f"Conjunto de prueba: {X_test.shape}")
except FileNotFoundError:
    print("No se encontraron los datos")
    exit()

# de indices a caracteres para visualizacion
abc = [chr(i) for i in range(ord('A'), ord('Z')+1)]


# ------------------------------- ARQUITECTURA DE RED ---------------------------------------
learning_rate = 0.001 # estandar
modelo = Sequential()   # arquitectura secuencial

# Capa de entrada implicita + primera capa oculta (2500 neuronas de entrada -> 512 ocultas)
# Dense: cada una de 512 neuronas estan conectadas con cada entrada anterior
# relu: filtro matematico, activa neuronas solo si se detecta información util
modelo.add(Dense(512, activation='relu', input_shape=(2500,)))

# Capas ocultas intermedias, se reduce numero de neuronas, cada una combina rasgos basicos detectados
# se maneiene la misma funcion de activacion
modelo.add(Dense(256, activation='relu'))
modelo.add(Dense(128, activation='relu'))

# Capa de salida (26 clases correspondientes a las letras A-Z)
# para la funcion de activacion softmax toma cada probabilidad de las 26, notmaliza y la que tiene el porcentaje más alto es la letra
modelo.add(Dense(26, activation='softmax'))

# optimizador Adam (Adaptative Moment Estimation)
# actualiza pesos de neuronas para reducir error de modelo
optimizador = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compilar con entropia cruzada categorica
# categorical_crossentropy: penaliza red cuando se equivoca
# accuracy: calcula porcentace de aciertos
modelo.compile(
    optimizer=optimizador,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# resumen estructural de la red
modelo.summary()


# ------------------------ ENTRENAMIENTO -------------------------------------
# 20% interno de X_train para la validacion en tiempo real por epoca
# epochs: cantidad de veces que red revisa, analiza y aprende del conjunto de entrenamiento
# batch_size: cantidad de imagenes que procesa red antes de actualizas pesos
# validation_split: separa 20% para probarlos al final de cada epoca, mide en tiempo real que tan bien va
historial = modelo.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=128,
    validation_split=0.2
)


# -------------------------- EVALUACION DE PRUEBA --------------------------
# toma imagenes de prueba y sus etiquetas, las pasa por la red, genera predicciones y devuelve valor de perdida e exactitud
test_loss, test_accuracy = modelo.evaluate(X_test, y_test)

print("\n" + "*"*50)
print(f"Perdida en conjunto de prueba: {test_loss:.4f}")
print(f"Exactitud en el conjunto de prueba: {test_accuracy*100:.2f}%")
print("*"*50 + "\n")

# -------------------------- GRAFICAS DE DESEMPEÑO --------------------------------
plt.figure(figsize=(12, 5))

# Evolucion de Accuracy
plt.subplot(1, 2, 1)
plt.plot(historial.history['accuracy'], label='Exactitud entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Exactitud validación')
plt.title('Evolución de exactitud (accuracy)')
plt.xlabel('Epoca')
plt.ylabel('Exactitud')
plt.legend()
plt.grid(True)

# Evolución de Loss
plt.subplot(1, 2, 2)
plt.plot(historial.history['loss'], label='Perdida entrenamiento')
plt.plot(historial.history['val_loss'], label='Pérdida validación')
plt.title('Evolución perdida (loss)')
plt.xlabel('Epoca')
plt.ylabel('Perdida')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------------- INFERENCIAS Y AUDITORIA VISUAL-------------------------------------
# pasa imagenes test por red entrenada, devuelce matriz, cada fila representa una img y tiene 26 prob
predicciones = modelo.predict(X_test)
# busca la que tuvo la probabilidad mas alta y asigna esa etiqueta
predicciones_clases = np.argmax(predicciones, axis=1)

# 15 indices al azar del conjunto de test
total_muestras = X_test.shape[0]
indices_aleatorios = np.random.choice(total_muestras, size=15, replace=False)

plt.figure(figsize=(12, 7))
for i, indice in enumerate(indices_aleatorios):
    plt.subplot(3, 5, i + 1)
    
    # Recontruir imagen en matriz (50x50) para grafica
    imagen = X_test[indice].reshape(50, 50)
    plt.imshow(imagen, cmap='gray')
    
    # Decodificacion de one hot
    indice_real = np.argmax(y_test[indice]) if len(y_test.shape) > 1 else int(y_test[indice])
    letra_real = abc[indice_real]
    letra_pred = abc[predicciones_clases[indice]]
    
    # verde acerto, rojo fallo
    color_titulo = 'green' if letra_real == letra_pred else 'red'
    plt.title(f"Real: {letra_real} | Pred: {letra_pred}", color=color_titulo, fontsize=10)
    plt.axis('off')

plt.suptitle("Inferencias en conjunto TEST")
plt.tight_layout()
plt.show()

# ----------------- GUARDAR MODELO ---------------------------
modelo.save("modelo_abecedario.keras")
print("Modelo guardado")