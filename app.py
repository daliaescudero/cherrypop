from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import tensorflow as tf

# Desactivar advertencias de TensorFlow
tf.get_logger().setLevel('ERROR')

app = Flask(__name__, static_folder="static")

# Cargar el modelo y las etiquetas
try:
    model = load_model("keras_model.h5", compile=False)
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

try:
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print("Etiquetas cargadas correctamente.")
except Exception as e:
    print(f"Error al cargar las etiquetas: {e}")
    class_names = []

# Crear directorio de imágenes si no existe
if not os.path.exists("static/images"):
    os.makedirs("static/images")

# Diccionario de consejos
SKIN_ADVICE = {
    "Piel Mixta": "Utilizar limpiadores que respeten el pH de la piel, eliminando impurezas sin resecar las zonas secas ni provocar exceso de sebo en la zona T.",
    "Piel Grasa": "Aunque sea grasa, esta piel necesita hidratación. Los productos en gel y sin aceites ayudan a equilibrar la producción de sebo.",
    "Piel Normal": "Utiliza productos de higiene suaves y eficaces que no eliminen la barrera protectora de la piel, como jabones y geles delicados con el microbioma, que fijen la hidratación natural de la piel y refuercen sus defensas naturales.",
    "Piel Sensible": "Usar cremas con protección solar todo el año, incluso en invierno, para proteger la piel sensible de los rayos UV.",
    "Piel Seca": "Aplicar cremas nutritivas dos veces al día y realizar una exfoliación semanal."
}

def get_skin_advice(skin_type):
    # Normalizar el texto para evitar errores de coincidencia
    skin_type = skin_type.strip()
    advice = SKIN_ADVICE.get(skin_type)
    if advice:
        return advice
    else:
        # Mensaje claro si no se encuentra el consejo
        return f"No se encontró consejo para el tipo de piel: {skin_type}."

# Función para procesar la imagen y hacer la predicción
def process_image(image_path):
    try:
        # Abrir y procesar la imagen
        image = Image.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Hacer la predicción
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()

        return {"class": class_name}
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
        except Exception as e:
            return jsonify({"error": "Error al guardar la imagen."}), 500

        # Procesar la imagen
        result = process_image(filepath)
        if "error" in result:
            return jsonify({"error": result["error"]})

        # Obtener el consejo según el tipo de piel
        skin_type = result["class"]
        advice = get_skin_advice(skin_type)

        # Enviar la respuesta
        return jsonify({
            "class": skin_type,
            "advice": advice
        })

    return render_template("index.html")

if __name__ == "__main__":
    # Asignar puerto dinámico para producción
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)




