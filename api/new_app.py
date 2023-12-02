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
      
@app.route('/predict/<model_type>', methods=['POST'])
def predict_digit(model_type):
    data = request.get_json()

    image_data = data['image']

    predictions = []
    
    image_array = np.array(json.loads(image_data))
    preprocessed_image = preprocess_data(image_array)

    # Dynamically load the first model in the 'models/' folder
    model_files = os.listdir('models/')
    model_files = [file for file in model_files if file.endswith('.joblib')]
    supported_model_types = ['svm', 'tree', 'lr']
    

    if model_type == 'svm':
        # return_msg = { "model_type" : f"You have passed {model_type}"}
        # return return_msg

        first_model_file = model_files[3]
        print(first_model_file)
        first_model_path = f"models/{first_model_file}"
        best_model = joblib.load(first_model_path)

        # Use the loaded model for prediction
        predicted_digit = best_model.predict(preprocessed_image.reshape(1, -1))[0]

    response = {
        "predictions": int(predicted_digit)
    }

    return jsonify(response)