import joblib
import json
import os
# print('==>',os.getcwd())
from flask import Flask, request, jsonify
from utils import preprocess_data
import numpy as np

# from markupsafe import escape

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict_digit():
    data = request.get_json()
    image_array = data['image']

    image_array = np.array(json.loads(image_array))
    preprocessed_image = preprocess_data(image_array)

    # Dynamically load the first model in the 'models/' folder
    model_files = os.listdir('models/')
    model_files = [file for file in model_files if file.endswith('.joblib')]

    if not model_files:
        raise FileNotFoundError("No model files found in the 'models/' folder")

    first_model_file = model_files[1] 
    # first_model_file = '/Users/sanjib/Desktop/IITJ/Classes/Sem-2/ML-Ops/Labs/digit-classifications/models/decision_tree_max_depth:10.joblib'
    first_model_path = f"models/{first_model_file}"
    best_model = joblib.load(first_model_path)

    # Use the loaded model for prediction
    predicted_digit = best_model.predict(preprocessed_image.reshape(1, -1))[0]

    response = {
        "predicted_digit": int(predicted_digit)
    }

    # return jsonify(response)
    return str(predicted_digit)