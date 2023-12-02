# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm
import pandas as pd
from utils import preprocess_data, read_digits, predict_and_eval, train_test_dev_split, tune_hparams, generate_hyperparameter_combinations
from joblib import dump, load
import sys
import numpy as np
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

# Logistic Regression parameters
lr_params = models["logistic_regression"]
lr_h_params_combinations = generate_hyperparameter_combinations(lr_params)


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
    # print('Iteration:',i)
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            train_size = 1- test_size - dev_size
            # 3. Data splitting -- to create train and test sets                
            X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)
            # 4. Data preprocessing
            X_train = preprocess_data(X_train)
            X_test = preprocess_data(X_test)
            X_dev = preprocess_data(X_dev)
        
            # Hyperparameter tuning and evaluation for SVM
            model = 'svm'
            best_svm_hparams, best_svm_model_path, best_svm_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, 
                                                                                    svm_h_params_combinations, 
                                                                                    model_type = model)
            best_svm_model = load(best_svm_model_path)
            svm_test_acc = predict_and_eval(best_svm_model, X_test, y_test)
            svm_train_acc = predict_and_eval(best_svm_model, X_train, y_train)
            svm_dev_acc = best_svm_accuracy

            # Hyperparameter tuning and evaluation for Decision Tree
            model = 'decision_tree'
            best_dt_hparams, best_dt_model_path, best_dt_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, 
                                                                                dt_h_params_combinations, 
                                                                                model_type=model)
            
            best_dt_model = load(best_dt_model_path)
            dt_test_acc = predict_and_eval(best_dt_model, X_test, y_test)
            dt_train_acc = predict_and_eval(best_dt_model, X_train, y_train)
            dt_dev_acc = best_dt_accuracy
            
            for solver in lr_params["solver"]:
                # Hyperparameter tuning and evaluation for Logistic Regression with each solver
                model = 'logistic_regression'
                accuracies = []
                best_lr_hparams, best_lr_model_path, best_lr_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev,
                                                                                    lr_h_params_combinations,
                                                                                    model_type=model, solver=solver)
                best_lr_model = load(best_lr_model_path)
                lr_test_acc = predict_and_eval(best_lr_model, X_test, y_test)
                lr_train_acc = predict_and_eval(best_lr_model, X_train, y_train)
                lr_dev_acc = best_lr_accuracy

                results_df = pd.concat([results_df, pd.DataFrame({
                    "Model Name": ["SVM", "Tree", f"Logistic Regression ({solver})"],
                    "Train Size": [train_size, train_size, train_size],
                    "Dev Size": [dev_size, dev_size, dev_size],
                    "Test Size": [test_size, test_size, test_size],
                    "Train Accuracy": [svm_train_acc, dt_train_acc, lr_train_acc],
                    "Dev Accuracy": [svm_dev_acc, dt_dev_acc, lr_dev_acc],
                    "Test Accuracy": [svm_test_acc, dt_test_acc, lr_test_acc]
                    })], ignore_index=True)
                
                lr_results_df = results_df[results_df["Model Name"].str.contains("Logistic Regression")]

# Reset the index of the new DataFrame
lr_results_df.reset_index(drop=True, inplace=True)

# Display the new DataFrame
print(lr_results_df)
                

# Display the results dataframe
# print(results_df)