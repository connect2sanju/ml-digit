import joblib
from flask import Flask, request, jsonify
from ..utils import preprocess_data
import os
import numpy as np
import json

from markupsafe import escape

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict_digit():
    data = request.get_json()

    # Check if 'images' key is present in the JSON data
    if 'images' not in data:
        return jsonify({"error": "Images key not found"}), 400

    images_data = data['images']

    if not isinstance(images_data, list):
        return jsonify({"error": "Images should be a list"}), 400

    predictions = []

    for i, image_data in enumerate(images_data, start=1):
        # Check if each image in the list has 'image' key
        if 'image' not in image_data:
            return jsonify({"error": f"Image{i} key not found"}), 400

        image_array = np.array(json.loads(image_data['image']))
        preprocessed_image = preprocess_data(image_array)

        # Dynamically load the first model in the 'models/' folder
        model_files = os.listdir('models/')
        model_files = [file for file in model_files if file.endswith('.joblib')]

        if not model_files:
            raise FileNotFoundError("No model files found in the 'models/' folder")

        first_model_file = model_files[0]
        first_model_path = f"models/{first_model_file}"
        best_model = joblib.load(first_model_path)

        # Use the loaded model for prediction
        predicted_digit = best_model.predict(preprocessed_image.reshape(1, -1))[0]

        predictions.append({"image": i, "predicted_digit": int(predicted_digit)})

    response = {
        "predictions": predictions
    }

    return jsonify(response)