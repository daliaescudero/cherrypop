from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

app = Flask(__name__)

# Cargar el modelo y las etiquetas
try:
    model = load_model("keras_model.h5", compile=False)
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

try:
    with open("labels.txt", "r") as f:
        class_names = f.readlines()
    print("Etiquetas cargadas correctamente.")
except Exception as e:
    print(f"Error al cargar las etiquetas: {e}")
    class_names = []

# Asegurarse de que el directorio de imágenes existe
if not os.path.exists("static/images"):
    os.makedirs("static/images")

# Función para procesar la imagen y hacer la predicción
def process_image(image_path):
    try:
        # Abrir la imagen
        image = Image.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Hacer la predicción
        prediction = model.predict(data)
        print(f"Predicción del modelo: {prediction}")  # Ver los valores de la predicción
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]
        confidence_score = float(confidence_score)

        return {"class": class_name, "confidence": confidence_score}
    except Exception as e:
        print(f"Error en el procesamiento de la imagen: {e}")
        return {"error": str(e)}

# Ruta principal
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No se proporcionó una imagen."}), 400

        # Guardar la imagen
        filepath = os.path.join("static", "images", file.filename)
        try:
            file.save(filepath)
            print(f"Archivo guardado correctamente en: {filepath}")
        except Exception as e:
            print(f"Error al guardar el archivo: {e}")
            return jsonify({"error": "Error al guardar la imagen."}), 500

        # Procesar la imagen
        result = process_image(filepath)
        print(f"Resultado de la predicción: {result}")  # Mostrar el resultado de la predicción
        return jsonify(result)

    # Renderizar la página HTML
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Heroku asigna un puerto dinámico
    app.run(debug=True, port=port)

