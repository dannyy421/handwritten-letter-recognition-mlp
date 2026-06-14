# Clasificador de letras mediante red neuronal y vision artificial
El proyecto implementa sistema de reconocimiento de caracteres para clasificar letras mayúsculas manuscritas del abecedario. Abarca desde el procesamiento de dataset, entrenamiento de red neuronal (perceptrón multicapa MLP) en TensorFlow/Keras, hasta inferencia y validación en tiempo real mediante interfaz de usuario con webcam.

## Estructura de repositorio

El repositorio se organiza bajo la siguiente arquitectura de archivos y directorios:

```text
├── DATASET/                  # Directorio con las muestras 
├── Archivos_X_y/             # Matrices empaquetadas (.npy) para entrenamiento/prueba
├── modelo_abecedario.keras   # Modelo de red neuronal guardado y optimizado
├── preparar_dataset.py       # Script de binarización, fusión y partición de datos
├── red_neuronal.py           # Arquitectura, entrenamiento y gráficas del modelo
├── prueba_modelo_abecedario.py # Script probando el modelo con los datos test
└── reconocimiento_camara.py  # Aplicación de inferencia en tiempo real con interfaz grafica
```

## Descripción
### 1. Preprocesamiento e integracion
* Binarizacion
* Consolidación de dataset grupal
* Aplanado y normalizacion
* Partición en datos de prueba y de entrenamiento

### 2. Arquitectura y entrenamiento
* **Arquitectura de la red:** 
    * Capa de entrada: implícita por configuración de tensor de receptor con dimensiones (2500,)
    * Capas ocultas: tres capas conectadas de 512, 256 y 128 neuronas respectivamente con funciones de activación RELU
    * Capas de salida: capa de 26 neuronas cada una correspondiente a una letra del abecedario. Activación por SOFTMAX
  * **Compilación y entrenamiento:** Optimizador ADAM con tasa de aprendizaje de 0.001 y función de pérdida `Categórical Crossentropy`. Se ejecutan 25 épocas con tamaño de 128 muestras y división de 20% para validación en tiempo real.
  * **Resultados:** Genera curvas de historial de pérdida y exactitud, evualuando el modelo contra el conjunto de prueba para luego exportar el archivo `modelo_abecedario.keras`

## 3. Pruebas
  Mediante el archivo `prueba_modelo_abecedario.py`realiza pruebas de el modelo entrenado sobre un conjunto aleatorio de los datos separados para la prueba, donde se escpecifíca su etiqueta real, su etiqueta que predijo el modelo y si acertó o no la predicción.

## 4. Implementación en tiempo real: 
  La interfaz de uzuarui con la libreria `Tkinter` e integrada con `ÒpenCV`:
  * Zona de enfoque: al iniciar el programa se detecta la resolución de la camara y calcula un cuadrado centrado de 250x250 pixeles
  * Procesamiento de inferencias: Casa 15 ms se captura una región de interes delimitada, la procesa como se hizo con el dataset (escala de grises, redimension a 50x50 pixeles, binarización inversa con Otsu) y el modelo predice su etiqueta.
  * Visualización de resultado: se implime la clasificación en tiempo real directamente sobre la pantalla

## Requisistos de sistema y ejecución
  Son necesarias las siguientes librerías:
   ```bash
   pip install tensorflow opencv-python pillow numpy matplotlib scikit-learn
   ```
### Flujo de ejecución
1. Preparar y generar matrices (descomentar las funciones de procesamiento) ejecutando:
    ```bash
   python preparar_dataset.py
   ```
2. Entrenar red neuronal
   ```bash
   python red_neuronal.py
   ```
3. Probar el modelo con dataset de prueba
    ```bash
   python prueba_modelo_abecedario.py
   ```
4. Ejecutar interfaz
   ```bash
   python reconocimiento_camara.py
   ```
  
