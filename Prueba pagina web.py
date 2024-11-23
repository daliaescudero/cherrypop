from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

app = Flask(__name__)

# Carga del modelo y las etiquetas
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()


def process_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        # Asegurarse de que el valor de 'confidence_score' es serializable como float
        confidence_score = float(confidence_score)

        return {"class": class_name, "confidence": confidence_score}
    except Exception as e:
        return {"error": str(e)}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No se proporcionó una imagen."}), 400

        # Guardar imagen subida
        filepath = os.path.join("static/images", file.filename)
        file.save(filepath)

        # Procesar imagen
        result = process_image(filepath)
        return jsonify(result)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
