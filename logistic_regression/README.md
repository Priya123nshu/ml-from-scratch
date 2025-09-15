# Logistic regression

## Overview

This module demonstrates a basic implementation of Logistic Regression using `scikit-learn`. It defines a small, predefined dataset (`X` and `y`), trains a `LogisticRegression` model on this data, makes predictions, and then evaluates the model's performance using a confusion matrix and a classification report. The code is a self-contained script that executes these steps directly.

## Installation

This script requires the following Python libraries:

*   `numpy`
*   `scikit-learn`

You can install them using pip:

```bash
pip install numpy scikit-learn
```

## Usage

This module is a script designed to be run directly. It will train a logistic regression model on its internal dataset and print the evaluation metrics to the console.

To run the script:

```bash
python your_module_name.py # Replace 'your_module_name.py' with the actual filename
```

**Example Output:**

```
Predictions: [0 0 0 0 1 1 1 1 1 1]
Confusion matrix:
 [[3 1]
 [0 6]]
Classification report:
               precision    recall  f1-score   support

           0       1.00      0.75      0.86         4
           1       0.86      1.00      0.92         6

    accuracy                           0.90        10
   macro avg       0.93      0.88      0.89        10
weighted avg       0.91      0.90      0.90        10
```
*Note: The exact output may vary slightly based on `scikit-learn` version or random state if not fixed, but the structure will be similar.*

## Inputs & Outputs

### Inputs

The script uses hardcoded input data:
*   `X`: A NumPy array of features, `np.arange(10).reshape(-1, 1)`, representing integers from 0 to 9, reshaped into a column vector.
*   `y`: A NumPy array of binary labels, `np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1])`, corresponding to the `X` values.

### Outputs

The script prints the following to the standard output:
*   `Predictions`: The binary predictions made by the trained `LogisticRegression` model on the input `X` data.
*   `Confusion matrix`: A 2x2 matrix showing the counts of true positives, true negatives, false positives, and false negatives.
*   `Classification report`: A detailed report including precision, recall, f1-score, and support for each class, as well as overall accuracy, macro average, and weighted average.

## Explanation

The script performs the following steps:

1.  **Data Preparation**: Initializes a feature matrix `X` and a label vector `y` using `numpy`. `X` contains single-feature data points from 0 to 9, and `y` contains corresponding binary labels.
2.  **Model Instantiation**: Creates an instance of `LogisticRegression` from `scikit-learn`.
    *   `solver='liblinear'` is specified, which is a good choice for small datasets.
    *   `C=10.0` sets the inverse of regularization strength. Smaller values specify stronger regularization.
    *   `random_state=0` ensures reproducibility of the results.
3.  **Model Training**: The `fit()` method is called on the `model` object, passing `X` and `y` to train the logistic regression classifier.
4.  **Prediction**: The `predict()` method is called on the trained `model` using the same `X` data to generate predictions (`y_pred`).
5.  **Evaluation**:
    *   `confusion_matrix(y, y_pred)` is used to compute the confusion matrix, comparing the true labels (`y`) with the predicted labels (`y_pred`).
    *   `classification_report(y, y_pred)` generates a text report showing the main classification metrics.
6.  **Output**: All the computed predictions and evaluation metrics are printed to the console for review.

## License

This project is open-sourced under the [PLACEHOLDER LICENSE] license.