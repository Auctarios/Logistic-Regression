# Logistic-Regression
 A custom implementation of Logistic Regression.

---

## My_LogisticRegression
This class implements logistic regression for binary classification tasks.
It includes functionality for regularized **gradient descent optimization**, error tracking, and debugging output.

- ### Parameters
    - **n_iter (int):** The number of iterations, default: 500
    - **lr (float):** Learning rate, default: 0.8
    - **lambda_ (float):** L2 Regularization coef., default: 0
    - **´err´ (bool):** Calculates in-sample and out-of-sample cross-entropy loss if True, default: True
    - **debug (bool):** Prints some information while the model is working, default: False
    - **batch_size (int):** In each iteration, `batch_size` sized batch of data is used to calculate new weights, if 1, then stochastic gradient descent. default: 1

- ### Methods
    - **fit:** Fits the logistic regression model to the training data using batch gradient descent. The method updates the weights and bias of the model based on the input data. If `err` is True, it also calculates in-sample and out-of-sample errors at each iteration. If `debug` is True, it prints debugging information during training.
        - #### Parameters
            - **X:** Training data.
            - **y:** Target values.
            - **X_test:** Test data for out-of-sample error calculation.
            - **Y_test:** Test target values for out-of-sample error calculation.
    
    ---
    - **calculate_error:** Calculates the binary cross-entropy loss with optional L2 regularization.
        - #### Parameters
            - **y_pred:** Predicted values, outout of the logistic regression model.
            - **y:** Actual target values
        - #### Returns
            - The computed binary cross-entropy loss.
            
    ---

    - **predict_proba:** Makes predictions using the logistic regression model.
        - #### Parameters
            - **X:** Input data for which predictions are to be made.
        - #### Returns
            - Predicted probabilities.
            
    ---

    - **predict:** Makes predictions using the logistic regression model. Only returns 0 or 1.
        - #### Parameters
            - **X:** Input data for which predictions are to be made.
        - #### Returns
            - Predicted values (0 or 1).
            
---

## Some helper methods
- **sigmoid:** Calculates and returns the sigmoid of given input x
    - #### Parameters
        - **x (array):** Input value
    - #### Returns
        - Sigmoid of x
    ---
- **Batch_GD:** Performs a single batch gradient descent update on the model weights. This function computes the gradient of the loss function with respect to the model weights and bias for a given batch of data. It's designed to be used in iterative optimization algorithms for training machine learning models.
    - #### Parameters
        - **X:** The input features of the dataset.
        - **y:** The target values.
        - **predictions:** The model's predictions for the input features.
        - **weights:** The current weights of the model.
        - **lambda_:** The regularization parameter.
        - **batch_size:** The size of the batch to use for the gradient computation.
        - **iteration:** The current iteration number in the training process.
    - #### Returns
        - A tuple containing gradients of weights and bias.