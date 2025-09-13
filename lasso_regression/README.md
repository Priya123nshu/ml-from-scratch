# Lasso Regression

## Overview

This module demonstrates the application of Lasso (Least Absolute Shrinkage and Selection Operator) regression, a linear model with L1 regularization, to fit a non-linear dataset. It generates synthetic data, transforms the features using polynomial expansion, trains a Lasso model, makes predictions, and visualizes the results. The purpose is to illustrate how Lasso regression can model complex relationships while performing feature selection through regularization.

## Installation

This module requires the following Python libraries. You can install them using pip:

```bash
pip install numpy matplotlib scikit-learn
```

## Usage

To run this demonstration, save the provided code as a Python file (e.g., `lasso_demo.py`) and execute it from your terminal:

```bash
python lasso_demo.py
```

Upon execution, a `matplotlib` plot will be displayed, showing the generated scattered data points and the fitted Lasso regression curve.

## Inputs & Outputs

### Inputs

*   **Data**: The script generates its own synthetic dataset `X` and `y` internally.
    *   `X`: A 2D NumPy array of 40 samples, sorted, ranging from 0 to 5.
    *   `y`: A 1D NumPy array derived from `sin(X)` with added random noise.

### Outputs

*   **Plot**: A `matplotlib` plot window displaying:
    *   Blue scatter points representing the generated data (`X`, `y`).
    *   A red line representing the predictions from the fitted Lasso regression model across a continuous range of `X` values.

## Explanation

The code performs the following steps to demonstrate Lasso regression:

1.  **Generate Sample Data**:
    *   A reproducible random seed is set.
    *   `X` values are generated as 40 sorted random numbers between 0 and 5.
    *   `y` values are generated based on `sin(X)` with some Gaussian noise added, simulating a non-linear relationship.

2.  **Polynomial Feature Transformation**:
    *   `PolynomialFeatures(degree=3)` is used to create polynomial features up to degree 3 from the original `X` data. This allows a linear model to fit a non-linear curve. For each sample `x`, it generates `[1, x, x^2, x^3]`.

3.  **Lasso Regression Model Initialization and Fitting**:
    *   A `Lasso` regression model is initialized with:
        *   `alpha=0.01`: This is the regularization strength. A higher `alpha` increases the penalty for larger coefficients, potentially driving some coefficients to zero (feature selection).
        *   `max_iter=10000`: Sets the maximum number of iterations for the optimization algorithm.
    *   The `model.fit(X_poly, y)` method trains the Lasso model using the polynomial features (`X_poly`) and the target variable (`y`).

4.  **Prediction**:
    *   A new set of 100 evenly spaced `X` values (`X_fit`) is created between 0 and 5 to create a smooth curve for plotting.
    *   These `X_fit` values are also transformed into polynomial features using the same `poly` transformer.
    *   `model.predict(X_fit_poly)` uses the trained Lasso model to predict `y` values for these new polynomial features.

5.  **Plot Results**:
    *   `matplotlib.pyplot` is used to visualize the original data points (blue scatter) and the fitted Lasso regression curve (red line).
    *   A legend is added, and the plot is displayed.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details (if applicable).
(Placeholder - please choose and add an appropriate license.)