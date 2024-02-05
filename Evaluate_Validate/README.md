# Model Evaluation Metrics and Visualization

This folder contains a Python file with functions designed to evaluate the performance of a machine learning model, specifically a Logistic Regression model. These functions calculate various performance metrics and plot evaluation charts.

---

## Evaluation Functions

The Python file includes the following functions for model evaluation:

- **calculate_accuracy:** Computes the accuracy of the model's predictions.
- **calculate_precision:** Calculates the precision metric for the model.
- **calculate_recall:** Computes the recall metric for the model.
- **calculate_f1_score:** Calculates the F1 score, which is the harmonic mean of precision and recall.
- **plot_confusion_matrix:** Plots a confusion matrix to visualize the true positives, false positives, true negatives, and false negatives.
- **plot_roc_curve:** Plots the Receiver Operating Characteristic (ROC) curve.
- **calculate_roc_auc:** Calculates the Area Under the ROC Curve (AUC) metric.
- **eval_valid:** Runs all evaluation metrics and plots, and prints the results.

### Usage

To use these functions, import the Python file and call `eval_valid(y_pred, y)` where `y_pred` is the model's predicted probabilities and `y` is the actual target values. This function will print out the accuracy, precision, recall, and F1 score, plot the confusion matrix and ROC curve, and print out the AUC score.

### Visualizations

The evaluation includes visual representations to aid in interpreting the model's performance:

- The confusion matrix provides insight into the number of correct and incorrect predictions.
- The ROC curve and its corresponding AUC score illustrate the model's ability to discriminate between classes.

---

## Requirements

The functions in the file require the following libraries:

- matplotlib
- numpy
- seaborn
- scikit-learn

Ensure these libraries are installed in your Python environment before running the evaluation functions.

---

## Example Output

The `eval_valid` function will generate output similar to the following:

    Accuracy: 0.85
    Precision: 0.83
    Recall: 0.78
    F1-Score: 0.805
    Are Under Curve (AUC) 0.88

And will display the following plots:
- Confusion Matrix
- ROC Curve
---
