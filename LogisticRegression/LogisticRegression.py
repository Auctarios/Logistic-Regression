import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from timeit import default_timer as timer





def sigmoid(x):
    """
    Calculates and returns the sigmoid of given input x

    Parameters:
    x (float): input value

    Returns:
    array: sigmoid of x
    """
    return 1 / (1 + np.exp(-x))



def Batch_GD(X, y, predictions, weights, lambda_, batch_size, iteration):
    """
    Performs a single batch gradient descent update on the model weights.

    This function computes the gradient of the loss function with respect to
    the model weights (dw) and bias (db) for a given batch of data. It's
    designed to be used in iterative optimization algorithms for training
    machine learning models.

    Parameters:
    X (numpy.ndarray): The input features of the dataset.
                       Shape should be (n_samples, n_features).
    y (numpy.ndarray): The target values. Shape should be (n_samples,).
    predictions (numpy.ndarray): The model's predictions for the input features.
                                 Shape should be (n_samples,).
    weights (numpy.ndarray): The current weights of the model.
                             Shape should be (n_features,).
    lambda_ (float): The regularization parameter.
    batch_size (int): The size of the batch to use for the gradient computation.
    iteration (int): The current iteration number in the training process.

    Returns:
    tuple: A tuple containing gradients of weights (dw) and bias (db).
    """
    n_samples = X.shape[0]
    start_idx = (iteration * batch_size) % n_samples
    end_idx = min(start_idx + batch_size, n_samples)
    X_batch = X[start_idx:end_idx]
    y_batch = y[start_idx:end_idx]
    predictions_batch = predictions[start_idx:end_idx]
    dw = (1 / batch_size) * np.dot(X_batch.T, (predictions_batch - y_batch)) + (lambda_ * weights)
    db = (1 / batch_size) * np.sum(predictions_batch - y_batch)


    return dw, db

class My_LogisticRegression():
    """
    A custom implementation of Logistic Regression.

    This class implements logistic regression for binary classification tasks.
    It includes functionality for regularized gradient descent optimization,
    error tracking, and debugging output.

    Attributes:
    n_iter (int): Number of iterations for the gradient descent optimization.
    lr (float): Learning rate for gradient descent.
    lambda_ (float): Regularization parameter. A higher value specifies stronger regularization.
    weights (numpy.ndarray): Model weights after fitting the model.
    bias (float): Model bias (intercept) after fitting the model.
    errIN (list): In-sample error for each iteration, if error tracking is enabled.
    errOUT (list): Out-of-sample error for each iteration, if error tracking is enabled.
    debug (bool): If True, prints debug information during training.
    batch_size (int): Size of the batch used in batch gradient descent.

    Methods:
    fit: Fits the logistic regression model to the training data.
    calculate_error: Computes the binary cross-entropy loss, optionally with regularization.
    predict: Makes predictions using the logistic regression model.
    """
    def __init__(self, n_iter = 500, lr = 0.8, lambda_ = 0,
                 err = True, debug = False, batch_size: int = 1):
        self.n_iter = n_iter
        self.lr = lr
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None
        self.errIN = [] if err else None
        self.errOUT = [] if err else None
        self.debug = debug
        self.batch_size = batch_size



    def fit(self, X, y, X_test = None, Y_test = None):
        """
        Fits the logistic regression model to the training data using batch gradient descent.

        Parameters:
        X (numpy.ndarray): Training data, shape (n_samples, n_features).
        y (numpy.ndarray): Target values, shape (n_samples,).
        X_test (numpy.ndarray, optional): Test data for out-of-sample error calculation, shape (n_samples, n_features).
        Y_test (numpy.ndarray, optional): Test target values for out-of-sample error calculation, shape (n_samples,).

        The method updates the weights and bias of the model based on the input data.
        If `err` is True, it also calculates in-sample and out-of-sample errors at each iteration.
        If `debug` is True, it prints debugging information during training.
        """
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        if self.debug: print(f"Initial Weights: {self.weights}")
        self.bias = 0
        learning_rate = self.lr

        for i in range(self.n_iter):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_pred)

            if self.errIN is not None:
                errorIN = self.calculate_error(y_pred, y)
                self.errIN.append(errorIN)

                if X_test is not None:
                    out_lin_pred = np.dot(X_test, self.weights) + self.bias
                    out_y_pred = sigmoid(out_lin_pred)
                    errorOUT = self.calculate_error(out_y_pred, Y_test)
                    self.errOUT.append(errorOUT)

            dw, db = Batch_GD(X = X, y = y, predictions = y_pred, weights = self.weights,
                              lambda_ = self.lambda_, batch_size = self.batch_size,
                              iteration = i)

            if i % 10 == 0 and self.debug:
                print(f"Iteration {i}")
                print(f"First five predicted values (before sigmoid): {linear_pred[:5]}")
                print(f"First five sigmoid outputs: {y_pred[:5]}")
                print(f"Weights: {self.weights}")
                print(f"Bias: {self.bias}")
                if self.errIN is not None:
                    print(f"Error IN: {errorIN}")
                    if X_test is not None: print(f"Error OUT: {errorOUT}\n")

            self.weights = self.weights - learning_rate * dw
            self.bias = self.bias - learning_rate * db

    def calculate_error(self, y_pred, y):
        """
        Calculates the binary cross-entropy loss with optional L2 regularization.

        Parameters:
        y_pred (numpy.ndarray): Predicted values, output of the logistic regression model, shape (n_samples,).
        y (numpy.ndarray): Actual target values, shape (n_samples,).

        Returns:
        float: The computed binary cross-entropy loss.
        """
        reg_term = (self.lambda_ / 2) * np.sum(np.square(self.weights))
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) + reg_term

    def predict_proba(self, X):
        """
        Makes predictions using the logistic regression model.

        Parameters:
        X (numpy.ndarray): Input data for which predictions are to be made, shape (n_samples, n_features).

        Returns:
        numpy.ndarray: Predicted probabilities, shape (n_samples,).
        """
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        return y_pred
    

    def predict(self, X):
        """
        Makes predictions using the logistic regression model.

        Parameters:
        X (numpy.ndarray): Input data for which predictions are to be made, shape (n_samples, n_features).

        Returns:
        numpy.ndarray: Predicted values (0 or 1), shape (n_samples,).
        """
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        return [1 if i > 0.5 else 0 for i in y_pred]