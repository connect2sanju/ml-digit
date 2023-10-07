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

from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split, get_hyperparameter_combinations, tune_hparams
from joblib import dump, load
# 1. Get the dataset
X, y = read_digits()

# 2. Hyperparameter combinations
# 2.1. SVM
gamma_list = [0.001, 0.01, 0.1, 1]
C_list = [1, 10, 100, 1000]
svm_h_params={}
svm_h_params['gamma'] = gamma_list
svm_h_params['C'] = C_list
svm_h_params_combinations = get_hyperparameter_combinations(svm_h_params)


# 2.2. Decision Tree
max_depth_list = [5, 10, 15]
min_samples_split_list = [2, 5, 10]
min_samples_leaf_list = [1, 2, 4]
dt_h_params={}
dt_h_params['max_depth'] = max_depth_list
# dt_h_params['min_samples_split'] = min_samples_split_list
# dt_h_params['min_samples_leaf'] = min_samples_leaf_list
dt_h_params_combinations = get_hyperparameter_combinations(dt_h_params)

test_sizes =  [0.1, 0.2, 0.3, 0.45]
dev_sizes  =  [0.1, 0.2, 0.3, 0.45]
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
        best_svm_hparams, best_svm_model_path, best_svm_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, svm_h_params_combinations)
        best_svm_model = load(best_svm_model_path)
        svm_test_acc = predict_and_eval(best_svm_model, X_test, y_test)
        svm_train_acc = predict_and_eval(best_svm_model, X_train, y_train)
        svm_dev_acc = best_svm_accuracy

        # Hyperparameter tuning and evaluation for Decision Tree
        best_dt_hparams, best_dt_model_path, best_dt_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, dt_h_params_combinations, model_type='decision_tree')
        best_dt_model = load(best_dt_model_path)
        dt_test_acc = predict_and_eval(best_dt_model, X_test, y_test)
        dt_train_acc = predict_and_eval(best_dt_model, X_train, y_train)
        dt_dev_acc = best_dt_accuracy

        print("test_size={:.2f} dev_size={:.2f} train_size={:.2f}".format(test_size, dev_size, train_size))
        print("==>SVM - train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}".format(svm_train_acc, svm_dev_acc, svm_test_acc))
        print("==>Decision Tree - train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}".format(dt_train_acc, dt_dev_acc, dt_test_acc))