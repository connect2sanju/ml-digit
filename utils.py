from sklearn.model_selection import train_test_split
from sklearn import svm, datasets, metrics, linear_model
from sklearn import tree
from joblib import dump, load
import os
import itertools
from sklearn.preprocessing import StandardScaler

##
def generate_hyperparameter_combinations(model_params):
    hyperparameter_combinations = []

    model_combinations = []
    for param_name, param_values in model_params.items():
        model_combinations.append([(param_name, value) for value in param_values])

    model_hyperparameter_combinations = list(itertools.product(*model_combinations))

    for model_combination in model_hyperparameter_combinations:
        hyperparameters = {param_name: value for param_name, value in model_combination}
        hyperparameter_combinations.append(hyperparameters)

    return hyperparameter_combinations


def tune_hparams(X_train, y_train, X_dev, y_dev, h_params_combinations, model_type="svm", solver=None):
    best_accuracy = -1
    best_model_path = ""
    for h_params in h_params_combinations:
        if solver is not None and 'solver' in h_params:
            # If a solver is specified, skip iterations with different solvers
            if h_params['solver'] != solver:
                continue

        # 5. Model training
        model = train_model(X_train, y_train, h_params, model_type=model_type)
        # Predict the value of the digit on the dev subset
        cur_accuracy = predict_and_eval(model, X_dev, y_dev)
        if cur_accuracy > best_accuracy:
            best_accuracy = cur_accuracy
            best_hparams = h_params
            solver_name = best_hparams['solver'] if 'solver' in best_hparams else 'default'
            best_model_path = f"./models/m22aie234_lr_{solver_name}.joblib"
            best_model = model

    # save the best_model
    dump(best_model, best_model_path)
    return best_hparams, best_model_path, best_accuracy



def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y 

def preprocess_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    
    # Apply unit normalization using StandardScaler
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    return data_normalized
    # return data

# Split data into 50% train and 50% test subsets
def split_data(x, y, test_size, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=test_size,random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# train the model of choice with the model prameter
def train_model(x, y, model_params, model_type):
    if model_type == "svm":
        # Create a classifier: a support vector classifier
        clf = svm.SVC
    if model_type == "decision_tree":
        # Create a classifier: a decession tree classifier
        clf = tree.DecisionTreeClassifier
    if model_type == "logistic_regression":
        # Create a classifier: logistic regression
        clf = linear_model.LogisticRegression

    model = clf(**model_params)
    # train the model
    model.fit(x, y)
    return model


def train_test_dev_split(X, y, test_size, dev_size, random_state=42):
    X_train_dev, X_test, Y_train_Dev, y_test =  split_data(X, y, test_size=test_size, random_state=1)
    # print("train+dev = {} test = {}".format(len(Y_train_Dev),len(y_test)))
    
    X_train, X_dev, y_train, y_dev = split_data(X_train_dev, Y_train_Dev, dev_size/(1-test_size), random_state=1)
        
    return X_train, X_test, X_dev, y_train, y_test, y_dev

# Question 2:
def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    return metrics.accuracy_score(y_test, predicted)
