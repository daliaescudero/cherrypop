import tkinter as tk
from tkinter import filedialog, Label
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


def predict_image():
    
    file_path = filedialog.askopenfilename(
        filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png")]
    )
    if not file_path:
        return  

    try:
        image = Image.open(file_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        result_label.config(
            text=f"Clase: {class_name[2:]}\nPuntaje de confianza: {confidence_score:.2f}"
        )
    except Exception as e:
        result_label.config(text=f"Error al procesar la imagen: {e}")

model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

window = tk.Tk()
window.title("Clasificador de Imágenes")
window.geometry("500x300")

instruction_label = Label(window, text="Haz clic en el botón para cargar una imagen:")
instruction_label.pack(pady=10)

upload_button = tk.Button(window, text="Subir Imagen", command=predict_image)
upload_button.pack(pady=10)

result_label = Label(window, text="")
result_label.pack(pady=20)

window.mainloop()
