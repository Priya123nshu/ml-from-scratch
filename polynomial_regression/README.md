# Polynomial Regression

## Overview
This module demonstrates a basic implementation of polynomial regression. It generates synthetic data following a sine wave pattern with added noise, transforms the features using a polynomial function (degree 3), fits a linear regression model to these transformed features, and then visualizes the original data points alongside the fitted polynomial regression curve.

## Installation
This module requires the following Python libraries. You can install them using pip:

```bash
pip install numpy matplotlib scikit-learn
```

## Usage
This module is designed to be run as a standalone script. Execute the Python file directly to generate the data, perform the polynomial regression, and display the plot.

```bash
python your_module_name.py
```
*(Replace `your_module_name.py` with the actual name of your Python file).*

Upon execution, a matplotlib plot window will appear, showing the scattered original data points and a red line representing the polynomial regression fit.

## Inputs & Outputs
### Inputs
The script does not take external inputs (e.g., command-line arguments or function parameters). All data is generated internally:
*   `X`: An array of 40 sorted random numbers between 0 and 5, shaped as `(40, 1)`.
*   `y`: A corresponding array derived from `sin(X)` with added Gaussian noise, shaped as `(40,)`.

### Outputs
*   A `matplotlib` plot displaying:
    *   Blue scatter points: The original generated `(X, y)` data.
    *   Red line: The polynomial regression curve fitted to the data.

## Explanation
The script performs the following steps:

1.  **Data Generation**:
    *   Sets a random seed for reproducibility (`np.random.seed(0)`).
    *   Generates 40 `X` values uniformly distributed between 0 and 5, then sorts them.
    *   Generates `y` values by applying the sine function to `X` and adding normally distributed random noise to simulate real-world data.

2.  **Polynomial Feature Transformation**:
    *   An instance of `PolynomialFeatures` is created with `degree=3`. This transformer will convert a single feature `x` into multiple features: `[1, x, x^2, x^3]`.
    *   The `fit_transform` method is called on the generated `X` data, creating `X_poly` which now contains the polynomial features.

3.  **Linear Regression Model Fitting**:
    *   An instance of `LinearRegression` is initialized.
    *   The `fit` method is called with `X_poly` (the transformed features) and `y` (the target values) to train the model. The linear regression model then finds the optimal coefficients for the polynomial features.

4.  **Prediction for Visualization**:
    *   A new set of `X` values (`X_fit`) is created using `np.linspace` to generate 100 evenly spaced points between 0 and 5. These are used to draw a smooth regression curve.
    *   These `X_fit` values are transformed into polynomial features (`X_fit_poly`) using the *same* `PolynomialFeatures` instance that was used for training.
    *   The trained `model` then predicts the `y` values (`y_fit`) for these transformed points.

5.  **Plotting Results**:
    *   The original `(X, y)` data points are plotted as a scatter plot in blue.
    *   The predicted `(X_fit, y_fit)` values, representing the fitted polynomial regression curve, are plotted as a red line.
    *   A legend is displayed to differentiate between the data and the regression line.
    *   `plt.show()` displays the generated plot.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details (if applicable).
*(Note: A `LICENSE.md` file would need to be created separately if a specific license is desired. This is a placeholder.)*