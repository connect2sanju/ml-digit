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

# @app.route('/predict/<model_type>', methods=['POST'])
# def load_model(model_type):

#     supported_model_types = ['svm', 'tree', 'lr']
#     return_msg = { "model_type" : f"You have passed {model_type}"}

#     if model_type == 'svm':
#         return return_msg
    
#     elif model_type == 'tree':
#         return return_msg
    
#     elif model_type == 'lr':
#         return return_msg
#     else:
#         return { "model_type" : f"{model_type} model not supported. Supported models {supported_model_types}"}
    
    
@app.route('/predict/<model_type>', methods=['POST'])
def predict_digit(model_type):
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
        supported_model_types = ['svm', 'tree', 'lr']
        

        if model_type == 'svm':
            return_msg = { "model_type" : f"You have passed {model_type}"}
            # return return_msg

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