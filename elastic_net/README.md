# Elastic Net

## Overview

This Python script demonstrates the application of Elastic Net regression on a synthetic dataset. It generates a simple regression dataset, splits it into training and testing sets, trains an `ElasticNet` model, evaluates its performance using Mean Squared Error (MSE), and visualizes the regression line against the actual test data.

## Installation

The script requires the following Python libraries. You can install them using pip:

```bash
pip install numpy matplotlib scikit-learn
```

## Usage

This module is a self-contained script that generates data, trains a model, evaluates it, and plots the results. To run the demonstration, simply execute the Python file:

```bash
python your_script_name.py
```

Upon execution, the script will:
1. Print the Mean Squared Error (MSE) to the console.
2. Display a matplotlib plot showing the synthetic data points and the Elastic Net regression line.

The core logic of initializing and training the ElasticNet model is as follows:

```python
from sklearn.linear_model import ElasticNet

# Initialize ElasticNet with alpha (total regularization strength)
# and l1_ratio (mixing parameter between L1 and L2 regularization)
model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)

# Train the model on your training data
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
```

## Inputs & Outputs

### Inputs

The script uses hardcoded parameters for data generation and model configuration:

*   **Data Generation (`make_regression`)**:
    *   `n_samples`: 200 (number of data points)
    *   `n_features`: 1 (number of independent variables)
    *   `noise`: 20 (standard deviation of the Gaussian noise applied to the output)
    *   `random_state`: 42 (for reproducibility)
*   **Data Splitting (`train_test_split`)**:
    *   `test_size`: 0.3 (30% of data for testing, 70% for training)
    *   `random_state`: 42 (for reproducibility)
*   **ElasticNet Model (`ElasticNet`)**:
    *   `alpha`: 1.0 (constant that multiplies the L1 and L2 terms)
    *   `l1_ratio`: 0.5 (the Elastic Net mixing parameter, with `0 <= l1_ratio <= 1`. `l1_ratio = 0` corresponds to L2 penalty, `l1_ratio = 1` to L1 penalty. `0.5` means an equal mix.)
    *   `random_state`: 42 (for reproducibility)

### Outputs

*   **Console Output**:
    *   The Mean Squared Error (MSE) calculated on the test set, printed to standard output.
        ```
        MSE: [a numerical value]
        ```
*   **Graphical Output**:
    *   A matplotlib plot titled "ElasticNet Regression", displaying:
        *   A scatter plot of the actual test data points (`X_test` vs `y_test`).
        *   A red line representing the Elastic Net model's predictions (`X_test` vs `y_pred`).

## Explanation

The script follows a standard machine learning workflow for regression tasks:

1.  **Data Generation**: It starts by creating a synthetic 1-dimensional regression dataset using `sklearn.datasets.make_regression`. This allows for a controlled environment to test the model.
2.  **Data Splitting**: The generated dataset is then divided into training and testing subsets using `sklearn.model_selection.train_test_split`. The training set is used to fit the model, and the unseen test set is used to evaluate its generalization performance.
3.  **Model Initialization and Training**: An `ElasticNet` model from `sklearn.linear_model` is instantiated. Elastic Net is a linear regression model that combines L1 (Lasso) and L2 (Ridge) regularization penalties. This combination allows for both feature selection (like Lasso) and handling of multicollinearity (like Ridge), often leading to better performance in certain scenarios. The `alpha` parameter controls the overall strength of regularization, and `l1_ratio` balances the mix between L1 and L2 penalties. The model is then `fit` to the `X_train` and `y_train` data.
4.  **Prediction**: After training, the model makes predictions (`y_pred`) on the `X_test` data.
5.  **Evaluation**: The model's performance is quantified by calculating the Mean Squared Error (MSE) between the `y_test` (actual values) and `y_pred` (predicted values) using `sklearn.metrics.mean_squared_error`. A lower MSE indicates a better fit.
6.  **Visualization**: Finally, `matplotlib.pyplot` is used to visualize the results. It plots the actual test data points as a scatter plot and overlays the regression line predicted by the Elastic Net model, providing a clear visual representation of how well the model fits the data.

## License

This project is open-sourced under a placeholder license. Please specify the desired license (e.g., MIT, Apache 2.0, GPL) here.