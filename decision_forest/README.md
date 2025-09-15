# Decision Forest

## Overview

This module provides a simple, self-contained Python script demonstrating the application of a Random Forest Classifier for a classification task. It uses the well-known Iris dataset, splitting it into training and testing sets, training a Random Forest model, and then evaluating its performance by calculating the accuracy score on the test set.

## Installation

This script relies on the `scikit-learn` library. You can install it using pip:

```bash
pip install scikit-learn
```

## Usage

This module is a standalone script designed to be run directly. It will load the Iris dataset, train a Random Forest model, and print the accuracy score to the console.

To run the script, save the code as a `.py` file (e.g., `decision_forest_example.py`) and execute it from your terminal:

```bash
python decision_forest_example.py
```

**Example Output:**

```
Accuracy: 0.98
```

*(Note: The exact accuracy may vary slightly depending on scikit-learn version and environment, though with `random_state=42` it should be consistent for a given scikit-learn version.)*

## Inputs & Outputs

### Inputs

*   **Implicit Input**: The script internally loads the **Iris dataset** using `sklearn.datasets.load_iris()`. It does not take any command-line arguments or external file inputs.

### Outputs

*   **Console Output**: The script prints a single line to standard output (the console) showing the calculated accuracy score of the trained Random Forest model on the test set, formatted to two decimal places.
    Example: `Accuracy: 0.98`

## Explanation

The script implements the following steps to perform a classification task using a Random Forest:

1.  **Load Dataset**: The `load_iris()` function from `sklearn.datasets` is used to fetch the Iris dataset, which is a classic dataset for classification. The features (`X`) and target labels (`y`) are extracted.
2.  **Split Data**: The dataset is divided into training and testing sets using `train_test_split` from `sklearn.model_selection`. A `test_size` of 30% is used for the test set, and `random_state=42` ensures reproducibility of the split.
3.  **Initialize Model**: A `RandomForestClassifier` from `sklearn.ensemble` is initialized. It is configured with `n_estimators=100` (meaning 100 decision trees will be built in the forest) and `random_state=42` for reproducibility of the model training process.
4.  **Train Model**: The `fit()` method of the classifier is called with the training features (`X_train`) and training labels (`y_train`) to train the Random Forest model.
5.  **Make Predictions**: Once trained, the `predict()` method is used to make predictions on the unseen test features (`X_test`).
6.  **Evaluate Model**: The `accuracy_score()` function from `sklearn.metrics` is used to compare the predicted labels (`y_pred`) against the actual test labels (`y_test`), thereby calculating the overall accuracy of the model.
7.  **Print Result**: The calculated accuracy score is then printed to the console, formatted to two decimal places.

## License

This project is licensed under the [LICENSE NAME] - see the LICENSE.md file for details. (Placeholder)