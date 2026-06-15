import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

class interfaz:
    def __init__(self, window):
        self.window = window
        self.window.title("Reconocimiento de letras")
        self.window.geometry("950x650")
        self.recuadro_centrado = None

        # Cargar modelo entrenado
        try:
            self.modelo_letras = tf.keras.models.load_model("modelo_abecedario.keras")
            print("Modelo neuronal cargado")
        except Exception as e:
            print(f"No se pudo cargar el modelo {e}")
            messagebox.showerror("Error cargando 'modelo_abecedario.keras'")
            exit()

        # Mapeo del abecedario
        self.abc = [chr(i) for i in range(ord('A'), ord('Z')+1)]

        self.cap = cv2.VideoCapture(0)
        
        self.lbl_estado = tk.Label(window, text="Analizando letras en zona", 
                                   font=("Arial", 12, "bold"), fg="orange")
        self.lbl_estado.pack(pady=5)
        
        self.frame_pantallas = tk.Frame(window)
        self.frame_pantallas.pack(pady=10)

        self.lbl_video = tk.Label(window, bg="black")
        self.lbl_video.pack(pady=10)

        # CONTENEDOR DERECHO: etiqueta y la img procesada
        self.frame_preproceso = tk.Frame(self.frame_pantallas)
        self.frame_preproceso.grid(row=0, column=1, padx=10, sticky="n")
        self.lbl_tit_preproceso = tk.Label(self.frame_preproceso, text="Vista preprocesada", 
                                           font=("Arial", 10, "bold"), fg="blue")
        self.lbl_tit_preproceso.pack(pady=2)
        # imagen preprocesada (binarizada y escalada)
        self.lbl_video_preproceso = tk.Label(self.frame_preproceso, bg="gray", width=100, height=100)
        self.lbl_video_preproceso.pack(pady=5)

        self.btn_cerrar = tk.Button(window, text="Cerrar", font=("Arial", 12, "bold"), 
                                    bg="#d9534f", fg="white", command=self.cerrar)
        self.btn_cerrar.pack(pady=15)
        
        
        self.actualizar_camara()

    def clasificar_letra(self, frame, coords_rect):
        try:
            x1, y1 = coords_rect[0]
            x2, y2 = coords_rect[1]

            # region de interes
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                return None, None
            
            # PREPROCESAR igual que al dataset
            # grises
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # reescalar
            resized_roi = cv2.resize(gray_roi, (50,50))
            # binarizar e invertir
            _, thresh_roi = cv2.threshold(resized_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # normalizar
            normalized_roi = thresh_roi/255.0
            # aplanar
            flatten_roi = normalized_roi.reshape(1, 2500)

            # PREDECIR LETRA DE RED
            prediccion = self.modelo_letras.predict(flatten_roi, verbose=0)
            clase_ganadora = np.argmax(prediccion, axis=1)[0]
            probabilidad = prediccion[0][clase_ganadora]

            letra = None
            if probabilidad > 0.85:
                letra = self.abc[clase_ganadora]
                
            # Retorna tanto la letra detectada como la matriz binarizada para poder dibujarla
            return letra, thresh_roi

        except Exception as e:
            print(f"Error en prediccion {e}")
        return None, None

    def actualizar_camara(self):
        ret, frame = self.cap.read()
        if ret:
            # Centro de captura
            if self.recuadro_centrado is None:
                alto, ancho, _ = frame.shape
                tamano_cuadro = 250  # Tamaño del recuadro (250x250 px)
                
                # Calcular esquina superior izquierda (x1, y1) y la inferior derecha (x2, y2)
                x1 = int((ancho - tamano_cuadro) / 2)
                y1 = int((alto - tamano_cuadro) / 2)
                x2 = x1 + tamano_cuadro
                y2 = y1 + tamano_cuadro
                
                self.recuadro_centrado = np.array([[x1, y1], [x2, y2]], dtype=int)

            # Duplicado para procesar
            frame_procesamiento = frame.copy()

            # Rectangulo centrado
            cv2.rectangle(frame, tuple(self.recuadro_centrado[0]), tuple(self.recuadro_centrado[1]), (0, 255, 255), 2)

            # Clasificar la letra dentro 
            letra_detectada, img_preprocesada = self.clasificar_letra(frame_procesamiento, self.recuadro_centrado)

            # Interfaz visual sobre el video
            if letra_detectada:
                cv2.putText(frame, f"Letra: {letra_detectada}", (self.recuadro_centrado[0][0], self.recuadro_centrado[0][1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Esperando caracter...", (self.recuadro_centrado[0][0], self.recuadro_centrado[0][1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # --- RENDERIZAR LA IMAGEN PREPROCESADA  ---
            if img_preprocesada is not None:
                # Escalar a 150x150 para ver en interfaz
                img_mini_render = cv2.resize(img_preprocesada, (100, 100), interpolation=cv2.INTER_NEAREST)
                img_pil_pre = Image.fromarray(img_mini_render)
                imgtk_pre = ImageTk.PhotoImage(image=img_pil_pre)
                self.lbl_video_preproceso.imgtk = imgtk_pre
                self.lbl_video_preproceso.configure(image=imgtk_pre)

            # Renderizar en Tkinter
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lbl_video.imgtk = imgtk
            self.lbl_video.configure(image=imgtk)
            
        # Actualizacion cada 15ms
        self.window.after(15, self.actualizar_camara)

    def cerrar(self):
        # Liberar camara
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        self.window.destroy()



root = tk.Tk()
app = interfaz(root)
root.mainloop()