"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm
import pandas as pd
from utils import preprocess_data, read_digits, predict_and_eval, train_test_dev_split, tune_hparams, generate_hyperparameter_combinations
from joblib import dump, load
import sys
import json
import itertools

# 1. Get the dataset
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
    print('Iteration:',i)
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            train_size = 1- test_size - dev_size
            # 3. Data splitting -- to create train and test sets                
            X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)
            # 4. Data preprocessing
            X_train = preprocess_data(X_train)
            X_test = preprocess_data(X_test)
            X_dev = preprocess_data(X_dev)
            # print("test_size={:.2f} dev_size={:.2f} train_size={:.2f}".format(test_size, dev_size, train_size))
        
            # Hyperparameter tuning and evaluation for SVM
            model_svm = 'svm'
            best_svm_hparams, best_svm_model_path, best_svm_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, 
                                                                                    svm_h_params_combinations, 
                                                                                    model_type = model_svm)
            best_svm_model = load(best_svm_model_path)
            svm_test_acc = predict_and_eval(best_svm_model, X_test, y_test)
            svm_train_acc = predict_and_eval(best_svm_model, X_train, y_train)
            svm_dev_acc = best_svm_accuracy
            # print("==>SVM - train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}".format(svm_train_acc, svm_dev_acc, svm_test_acc))

            # Hyperparameter tuning and evaluation for Decision Tree
            model_tree = 'decision_tree'
            best_dt_hparams, best_dt_model_path, best_dt_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, 
                                                                                dt_h_params_combinations, 
                                                                                model_type=model_tree)
            best_dt_model = load(best_dt_model_path)
            dt_test_acc = predict_and_eval(best_dt_model, X_test, y_test)
            dt_train_acc = predict_and_eval(best_dt_model, X_train, y_train)
            dt_dev_acc = best_dt_accuracy

            results_df = pd.concat([results_df, pd.DataFrame({
                "Model Name": ["SVM", "Tree"],
                "Train Size": [train_size, train_size],
                "Dev Size": [dev_size, dev_size],
                "Test Size": [test_size, test_size],
                "Train Accuracy": [svm_train_acc, dt_train_acc],
                "Dev Accuracy": [svm_dev_acc, dt_dev_acc],
                "Test Accuracy": [svm_test_acc, dt_test_acc]
            })], ignore_index=True)

            # print("==>Decision Tree - train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}".format(dt_train_acc, dt_dev_acc, dt_test_acc))
        
print(results_df)