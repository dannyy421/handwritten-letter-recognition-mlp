import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import LabelEncoder


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


# BINARIZAR CARPETAS FALTANTES
origen = "Dataset_Robotica_A\Dataset_alumno01"   
destino = "Dataset_Robotica_A\Dataset_alumno40_bin" 
#binarizar(origen,destino)

# MUESTRA DE ALGUN DATASET
X = "Dataset_Robotica_A\Dataset_alumno40/X.npy"
y = "Dataset_Robotica_A\Dataset_alumno40/y.npy"
#muestra_npy(X,y)

# GENERAR DATASET GRUPAL
directorio = "Dataset_Robotica_A"
nombre_salida = "grupo"
#generar_dataset_grupal(directorio,nombre_salida)

# APLANADO
X_preparado = flatten_normalizado("Dataset_Robotica_A\X_grupo.npy")

# ETIQUETAS CONVERTIDAS
y_preparado = codificacion_etiquetas("Dataset_Robotica_A\y_grupo.npy")
np.save("Dataset_Robotica_A\X_preparado.npy", X_preparado)
np.save("Dataset_Robotica_A\y_preparado.npy", y_preparado)





