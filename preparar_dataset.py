import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def binarizar(carpeta_origen,carpeta_destino):
    lista_X = []
    lista_y = []
    letras = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    print("Proceso binarizacion\n")

    for letra in letras:
        ruta_carpeta_letra_origen = os.path.join(carpeta_origen, letra)
        
        # Verificar si carpeta de letra existe
        if not os.path.exists(ruta_carpeta_letra_origen):
            print(f"[Aviso] La carpeta para la letra {letra} no existe. Saltando...")
            continue
            
        # Crear la subcarpeta correspondiente en el destino 
        ruta_carpeta_letra_destino = os.path.join(carpeta_destino, letra)
        if not os.path.exists(ruta_carpeta_letra_destino):
            os.makedirs(ruta_carpeta_letra_destino)
            
        # Listar las imagenes (A_001.png, A_002.png...)
        archivos = [f for f in os.listdir(ruta_carpeta_letra_origen) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for archivo in archivos:
            ruta_imagen_origen = os.path.join(ruta_carpeta_letra_origen, archivo)
            
            # Lee en escala de grises
            img_gray = cv2.imread(ruta_imagen_origen, cv2.IMREAD_GRAYSCALE)
            
            if img_gray is None:
                print(f"   No se pudo cargar: {ruta_imagen_origen}")
                continue
                
            # Binarización con OTSU
            _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Guardar con nombre original
            ruta_imagen_destino = os.path.join(ruta_carpeta_letra_destino, archivo)
            cv2.imwrite(ruta_imagen_destino, img_bin)
            
            # almacenar image y su etiqueta
            lista_X.append(img_bin)
            lista_y.append(letra)
            
        print(f"-> Carpeta [{letra}] procesada ({len(archivos)} imagenes).")

    # Generar .npy dentro de la nueva carpeta Dataset_alumno_bin
    if lista_X:
        X = np.array(lista_X, dtype=np.uint8)
        y = np.array(lista_y, dtype=str)

        np.save(os.path.join(carpeta_destino, "X.npy"), X)
        np.save(os.path.join(carpeta_destino, "y.npy"), y)
        
        print("Archivos 'X.npy' y 'y.npy' actualizados")
    else:
        print("\n[Error] No se pudo procesar ninguna imagen.")

def muestra_npy(archivo_X,archivo_y):

    #cargar achivos 
    try:
        #X = np.load("Dataset_alumno40/X.npy")
        #y = np.load("Dataset_alumno40/y.npy",allow_pickle=True)
        X = np.load(archivo_X)
        y = np.load(archivo_y,allow_pickle=True)
    except FileNotFoundError:
        print("Error")
        exit()

    # Muestra informacion del dataset
    print("\nInformacion del dataset")
    print(f"Cantidad total de imagenes: {X.shape[0]}")
    print(f"Tamaño de cada imagen: {X.shape[1]}x{X.shape[2]} píxeles")
    print(f"Cantidad total de etiquetas: {y.shape[0]}")

    # Verificar que coincidan las cantidades
    if X.shape[0] != y.shape[0]:
        print("Num de img y etiquetas NO coincide")
        exit()

    # Grafica muestra aleatoria de 9 imagenes
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.suptitle("Muestra aleatoria del dataset preprocesado", fontsize=16)

    # 9 indices al azar
    indices_aleatorios = random.sample(range(len(X)), min(9, len(X)))

    for i, ax in enumerate(axes.flat):
        if i < len(indices_aleatorios):
            idx = indices_aleatorios[i]
            
            # muestra en escala de grises
            ax.imshow(X[idx], cmap='gray')
            
            # pone su etiqueta "y" correspondiente
            ax.set_title(f"Etiqueta: {y[idx]}", fontsize=12)
            
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def generar_dataset_grupal(directorio, nombre_salida):
    # Cualquier carpeta que empiece por Dataset_alumno
    carpetas_alumnos = [
        f for f in os.listdir(directorio) 
        if os.path.isdir(os.path.join(directorio, f)) and f.startswith("Dataset_alumno")
    ]

    #ordenar en orden alfabetico
    carpetas_alumnos.sort()

    lista_X_global = []
    lista_y_global = []

    for carpeta in carpetas_alumnos:
        ruta_carpeta_alumno = os.path.join(directorio, carpeta)
        ruta_X = os.path.join(ruta_carpeta_alumno, "X.npy")
        ruta_y = os.path.join(ruta_carpeta_alumno, "y.npy")
        
        if os.path.exists(ruta_X) and os.path.exists(ruta_y):
            datos_X = np.load(ruta_X)
            datos_y = np.load(ruta_y, allow_pickle=True)
            
            lista_X_global.append(datos_X)
            lista_y_global.append(datos_y)
            print(f"-> Cargados con éxito datos de: {carpeta} ({len(datos_y)} imágenes)")
        else:
            print(f"[Aviso] Faltan archivos X.npy o y.npy en la carpeta: {carpeta}. Saltando...")

    # Concatena y guarda en el grupo
    if lista_X_global and lista_y_global:
        X_grupo = np.concatenate(lista_X_global, axis=0)
        y_grupo = np.concatenate(lista_y_global, axis=0)
        
        # Definir las rutas de guardado final al lado del script
        ruta_salida_X = os.path.join(directorio, f"X_{nombre_salida}.npy")
        ruta_salida_y = os.path.join(directorio, f"y_{nombre_salida}.npy")
        
        np.save(ruta_salida_X, X_grupo)
        np.save(ruta_salida_y, y_grupo)
        
        print(f"- Archivo 'X_grupo.npy' creado con dimensiones: {X_grupo.shape}")
        print(f"- Archivo 'y_grupo.npy' creado con dimensiones: {y_grupo.shape}")
        print(f"- Total de imágenes en el dataset del grupo: {len(y_grupo)}")

        return X_grupo, y_grupo
    else:
        print("\n[Error] No se encontraron datos para fusionar")
        return None, None

def flatten_normalizado(direccion_X):
    try:
        X = np.load(direccion_X)
        print(f"Forma original de X: {X.shape}")
    except FileNotFoundError:
        print("[Error]")
        exit()    
    
    num_muestras, alto, ancho = X.shape
    # aplanado
    X_flatten = X.reshape(num_muestras, alto*ancho)
    # normalizado
    X_final = X_flatten.astype('float32')/255
    return(X_final)

def codificacion_etiquetas(direccion_y):
    try:
        y = np.load(direccion_y, allow_pickle=True)
    except FileNotFoundError:
        print("Error")
        exit()
    # codificacion one hot
    label = LabelEncoder()  # label encoder convierte cada letra en numero (A=0, B=1, etc)
    y_numerico = label.fit_transform(y) #identifica cuales y cuantas clases unicas existen (fit)
                                        # sustituye cada texsto por su numero (transform)
    y_one_hot = np.eye(26)[y_numerico]  # matriz identidad, devuelve vector con ceros
                                        # y solamente un 1 al que le corresponde esa categoria
    return (y_one_hot)

def division_datos_train_y_test(X_preparado, y_preparado):
    X = np.load(X_preparado)
    y = np.load(y_preparado)

    print(f"Dataset original: {X.shape[0]} muestras.")

    # estratificacion
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,  # 20 a pruebas y 80 restante a entrenamiento
        random_state=1, # sincroniza la alatoriedad
        stratify=y  # proporcion de letras igual tanto en entrenamiento como en prueba
    )

    np.save("Archivos_X_y\X_train.npy", X_train)
    np.save("Archivos_X_y\y_train.npy", y_train)

    np.save("Archivos_X_y\X_test", X_test)
    np.save("Archivos_X_y\y_test", y_test)

    print(f"Conjunto de Entrenamiento (80%): {X_train.shape[0]} muestras")
    print(f"Conjunto de Prueba        (20%): {X_test.shape[0]} muestras")
    print(f"Total de muestras validadas:     {X_train.shape[0] + X_test.shape[0]}")

def validar_datos_train_y_test(y_train, y_test):
    y_train = np.load(y_train, allow_pickle=True)
    y_test = np.load(y_test, allow_pickle=True)
    # se regresan a numeros de 0 a 25 porque estaban en one hot
    if len(y_train.shape) > 1:
        y_train_labels = np.argmax(y_train, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
    else:
        y_train_labels = y_train
        y_test_labels = y_test

    # Abecedario para mostrar las letras en lugar de números
    abc = [chr(i) for i in range(ord('A'), ord('Z')+1)]

    # Contar frecuencia de cada letra
    clases_train, conteos_train = np.unique(y_train_labels, return_counts=True)
    clases_test, conteos_test = np.unique(y_test_labels, return_counts=True)

    print(f"{'Letra':<6} | {'Cant. Entrenamiento (80%)':<25} | {'Cant. Prueba (20%)':<20}")
    print("-" * 60)
    for i in range(26):
        letra = abc[i]
        c_train = conteos_train[i] if i in clases_train else 0
        c_test = conteos_test[i] if i in clases_test else 0
        print(f"  {letra:<4} | {c_train:<25} | {c_test:<20}")

    print("-" * 60)
    print(f"TOTAL  | {len(y_train_labels):<25} | {len(y_test_labels):<20}")

# BINARIZAR CARPETAS FALTANTES
origen = "DATASET\Dataset_alumno01"   
destino = "DATASET\Dataset_alumno40_bin" 
#binarizar(origen,destino)

# MUESTRA DE ALGUN DATASET
X = "Archivos_X_y\X_grupo.npy"
y = "Archivos_X_y\y_grupo.npy"
#muestra_npy(X,y)

# GENERAR DATASET GRUPAL
directorio = "DATASET"
nombre_salida = "grupo"
#generar_dataset_grupal(directorio,nombre_salida)

# APLANADO
#X_preparado = flatten_normalizado("Archivos_X_y\X_grupo.npy")

# ETIQUETAS CONVERTIDAS
#y_preparado = codificacion_etiquetas("Archivos_X_y\y_grupo.npy")
#np.save("Archivos_X_y\X_preparado.npy", X_preparado)
#np.save("Archivos_X_y\y_preparado.npy", y_preparado)

# DIVIDIR DATOS DE ENTRENAMIENTO Y PRUEBA
X_preparado = "Archivos_X_y\X_preparado.npy"
y_preparado = "Archivos_X_y\y_preparado.npy"
#division_datos_train_y_test(X_preparado,y_preparado)

# VERIFICAR DATOS DE ENTRENAMIENTO Y PRUEBA
y_train = "Archivos_X_y\y_train.npy"
y_test = "Archivos_X_y\y_test.npy"
validar_datos_train_y_test(y_train, y_test)




