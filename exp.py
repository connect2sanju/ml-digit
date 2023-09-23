# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Import datasets, classifiers and performance metrics
import itertools
from sklearn import metrics, svm
from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split, tune_hparams

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

# 1. Get the dataset
X, y = read_digits()

dev_size = [0.1, 0.2, 0.3] 
test_size = [0.1, 0.2, 0.3]
# [user data split as train_size: 0.7 dev_size: 0.1 test_size: 0.2 in all cases]

# dev_test_combinations = [{'test_size': test, 'dev_size': dev} for test, dev in itertools.product(dev_size, test_size)]

# 3. Data splitting -- to create train and test sets
# for dict_size in dev_test_combinations:
test_size = .2
dev_size = .1
train_size = .7

X_train, X_test, y_train, y_test, X_dev, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)

# 4. Data preprocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)
X_dev = preprocess_data(X_dev)

gamma_range = [0.001]
C_range = [0.1]


# Generate a list of dictionaries representing all combinations
param_combinations = [{'gamma': gamma, 'C': C} for gamma, C in itertools.product(gamma_range, C_range)]


# Hyperparameter tuning
train_acc, best_hparams, best_model, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, param_combinations)


# Model training
model = train_model(X_train, y_train, best_hparams, model_type="svm")

# Accuracy Evaluation
test_acc = predict_and_eval(model, X_test, y_test)

# Print all combinations
# print(f'test_size={test_size}, dev_size={dev_size}, train_size={train_size}, train_acc:{train_acc:.2f} dev_acc:{best_accuracy:.2f} test_acc: {test_acc:.2f}')
# print(f' Best params:{best_hparams}')

print('The number of total samples in the train dataset:', len(X)*train_size)
print('The number of total samples in the test dataset:', len(X)*test_size)
print('The number of total samples in the dev dataset:', len(X)*dev_size)

