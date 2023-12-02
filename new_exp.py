# Importing necessary libraries
from sklearn import metrics, svm
import pandas as pd
from utils import preprocess_data, read_digits, predict_and_eval, train_test_dev_split, tune_hparams, generate_hyperparameter_combinations
from joblib import dump, load
import json
import argparse

# Setting up argparse for command-line arguments
parser = argparse.ArgumentParser(description='Hyperparameter tuning and evaluation for SVM and Decision Tree models.')
parser.add_argument('--model', choices=['svm', 'decision_tree'], required=True, help='Specify the model (svm or decision_tree)')
parser.add_argument('--random_state', type=int, default=None, help='Specify the random_state value')
args = parser.parse_args()

# Get the dataset
X, y = read_digits()

# Load parameters from config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)

iter_count = config["iterations"]
test_sizes = config["test_sizes"]
dev_sizes = config["dev_sizes"]
models = config["models"]

# SVM parameters
svm_params = models["svm"]
svm_h_params_combinations = generate_hyperparameter_combinations(svm_params)

# Decision Tree parameters
dt_params = models["decision_tree"]
dt_h_params_combinations = generate_hyperparameter_combinations(dt_params)

results_df = pd.DataFrame(columns=[
    "Model Name",
    "Train Size",
    "Dev Size",
    "Test Size",
    "Train Accuracy",
    "Dev Accuracy",
    "Test Accuracy"
])

for i in range(iter_count):
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            train_size = 1 - test_size - dev_size
            # Data splitting -- to create train and test sets
            X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size, random_state=args.random_state)
            # Data preprocessing
            X_train = preprocess_data(X_train)
            X_test = preprocess_data(X_test)
            X_dev = preprocess_data(X_dev)

            # Hyperparameter tuning and evaluation
            if args.model == 'svm':
                model_type = 'svm'
                h_params_combinations = svm_h_params_combinations
            elif args.model == 'decision_tree':
                model_type = 'decision_tree'
                h_params_combinations = dt_h_params_combinations

            best_hparams, best_model_path, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev,
                                                                       h_params_combinations,
                                                                       model_type=model_type)
            best_model = load(best_model_path)
            test_acc = predict_and_eval(best_model, X_test, y_test)
            train_acc = predict_and_eval(best_model, X_train, y_train)
            dev_acc = best_accuracy

            results_df = pd.concat([results_df, pd.DataFrame({
                "Model Name": [args.model],
                "Train Size": [train_size],
                "Dev Size": [dev_size],
                "Test Size": [test_size],
                "Train Accuracy": [train_acc],
                "Dev Accuracy": [dev_acc],
                "Test Accuracy": [test_acc]
            })], ignore_index=True)

print(results_df.head(10))
