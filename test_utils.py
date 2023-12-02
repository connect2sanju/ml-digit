from utils import generate_hyperparameter_combinations, train_test_dev_split,read_digits, tune_hparams, preprocess_data
import os
import unittest
from joblib import load
from sklearn import linear_model
import glob


def test_loaded_model_is_logistic_regression():
    # Replace with your actual roll number and solver name used in the test case
    roll_number = "m22aie234"
    solver_name =  "lbfgs"
    model_path = f"./models/{roll_number}_lr_{solver_name}.joblib"

    # Load the model
    loaded_model = load(model_path)

    # Check if the loaded model is an instance of logistic regression
    assert (isinstance(loaded_model, linear_model.LogisticRegression))


def test_solver_name_matches_model_file_name():
    # Replace with your actual roll number and solver name used in the test case
    roll_number = "m22aie234"
    model_solver_list = ["liblinear", "lbfgs", "newton-cg", "default"]
    model_files = glob.glob(f"./models/{roll_number}_lr_*")
    print(model_files)

    for model_path in model_files:
        # Extract solver name from the model file name
        model_solver_name = model_path.split('_')[-1].split('.')[0]
        # # Check if the solver name in the model file name belongs to the given solver list
        assert model_solver_name in model_solver_list




 
# def test_for_hparam_cominations_count():
#     # a test case to check that all possible combinations of paramers are indeed generated
#     svm_param = {
#       "gamma": [0.001, 0.01],
#       "C": [1, 10]
#     }
#     h_params_combinations = generate_hyperparameter_combinations(svm_param)
    
#     assert len(h_params_combinations) == len(svm_param["gamma"]) * len(svm_param["C"])


# def create_dummy_hyperparameter():
#     svm_param = {
#       "gamma": [0.001, 0.01],
#       "C": [1]
#     }
#     h_params_combinations = generate_hyperparameter_combinations(svm_param)
#     return h_params_combinations


# def create_dummy_data():
#     X, y = read_digits()
    
#     X_train = X[:100,:,:]
#     y_train = y[:100]
#     X_dev = X[:50,:,:]
#     y_dev = y[:50]

#     X_train = preprocess_data(X_train)
#     X_dev = preprocess_data(X_dev)

#     return X_train, y_train, X_dev, y_dev


# def test_for_hparam_cominations_values():    
#     h_params_combinations = create_dummy_hyperparameter()
    
#     expected_param_combo_1 = {'gamma': 0.001, 'C': 1}
#     expected_param_combo_2 = {'gamma': 0.01, 'C': 1}

#     assert (expected_param_combo_1 in h_params_combinations) and (expected_param_combo_2 in h_params_combinations)

# def test_model_saving():
#     X_train, y_train, X_dev, y_dev = create_dummy_data()
#     h_params_combinations = create_dummy_hyperparameter()

#     _, best_model_path, _ = tune_hparams(X_train, y_train, X_dev, 
#         y_dev, h_params_combinations)   

#     assert os.path.exists(best_model_path)

# def test_data_splitting():
#     X, y = read_digits()
    
#     X = X[:100,:,:]
#     y = y[:100]
    
#     test_size = .1
#     dev_size = .6

#     X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)

#     assert (len(X_train) == 30) 
#     assert (len(X_test) == 10)
#     assert  ((len(X_dev) == 60))