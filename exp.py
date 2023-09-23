import itertools
from sklearn import metrics, svm
from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split, tune_hparams
from skimage.transform import resize
import numpy as np

# Load the digit dataset
X, y = read_digits()

# Define different image sizes
image_sizes = [(4, 4), (6, 6), (8, 8)]

# Define data split sizes
data_split_sizes = {
    'train_size': 0.7,
    'dev_size': 0.1,
    'test_size': 0.2
}

# Loop through image sizes
for size in image_sizes:
    for split_name, split_size in data_split_sizes.items():
        # Resize the images to the specified size
        X_resized = [resize(image, size) for image in X]

        # Convert the resized images to NumPy arrays
        X_resized = np.array(X_resized)

        # Split the data into train, dev, and test sets
        X_train, X_test, y_train, y_test, X_dev, y_dev = train_test_dev_split(X_resized, y, test_size=split_size, dev_size=split_size)

        # Preprocess the data
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

        # Print performance for the current image size and data split size
        print(f'Image size: {size[0]}x{size[1]} {split_name}: {split_size:.1f} '
              f'train_acc: {train_acc:.2f} dev_acc: {best_accuracy:.2f} test_acc: {test_acc:.2f}')

# Calculate and print the number of total samples
total_samples = len(X)
print(f'Total samples: {total_samples}')
