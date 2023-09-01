# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


# Data preprocessing, flatten the images
def preprocess_data(data):
    n_samples = len(data)
    data = digits.images.reshape((n_samples, -1))
    return data

# Split data into train test & validation subsets
def split_train_dev_test(X, y, test_size, dev_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=dev_size, random_state=1) 

    return X_train,y_train, X_val, y_val, X_test, y_test

# Train the model of choice with model parameter 
def train_model(x,y,model_param, model_type='svm'):
    model = svm.SVC(gamma=0.001)
    # Train the model
    model.fit(x, y)

    return model


def predict_and_eval(model, X_test, y_test):
# Predict the value of the digit on the test subset
    predicted = model.predict(X_test)

    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")


    # The ground truth and predicted lists
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    # For each cell in the confusion matrix, add the corresponding ground truths
    # and predictions to the lists
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )

    return predicted 


digits = datasets.load_digits()

data = preprocess_data(digits.images)

X_train,y_train, X_val, y_val, X_test, y_test = split_train_dev_test(data, digits.target, 0.2, 0.2)

model = train_model(X_train, y_train, {'gamma':0.001}, model_type='svm')
predicted = predict_and_eval(model, X_test, y_test)