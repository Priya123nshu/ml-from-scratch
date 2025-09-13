# Ridge Regression

## Overview

This module demonstrates a basic application of Ridge Regression with polynomial features. It generates synthetic 1D data, transforms it using polynomial features of degree 3, fits a Ridge regression model to this transformed data, and then visualizes the original data points alongside the fitted regression curve.

## Installation

The script requires the following Python libraries:
-   `numpy` for numerical operations.
-   `matplotlib` for plotting.
-   `scikit-learn` for Ridge regression and polynomial feature transformation.

You can install these dependencies using pip:

```bash
pip install numpy matplotlib scikit-learn
```

## Usage

To run this demonstration, save the provided code as a Python file (e.g., `ridge_regression_demo.py`) and execute it from your terminal:

```bash
python ridge_regression_demo.py
```

Upon execution, a `matplotlib` window will pop up, displaying the generated data points and the Ridge regression curve.

## Inputs & Outputs

**Inputs:**
The script does not take any external inputs. It internally generates its own synthetic dataset:
-   `X`: 40 data points, randomly generated between 0 and 5, and then sorted.
-   `y`: A target variable derived from `sin(X)` with added Gaussian noise.

**Outputs:**
-   A graphical plot (via `matplotlib.pyplot`) will be displayed, containing:
    -   Blue scatter points representing the generated synthetic data (`X` vs `y`).
    -   A red line representing the predictions from the fitted Ridge Regression model over a finer range of `X` values, illustrating the regression curve.

## Explanation

The script performs the following sequence of operations:

1.  **Data Generation**:
    -   `np.random.seed(0)` is set for reproducibility of the random data.
    -   `X` is created as 40 random numbers between 0 and 5, then sorted.
    -   `y` is computed by taking the sine of `X` and adding random noise (`np.random.randn`) to simulate real-world data with some variance.

2.  **Polynomial Feature Transformation**:
    -   An instance of `PolynomialFeatures(degree=3)` is initialized. This object is responsible for transforming the input features into polynomial features. For a single feature `x`, it will generate features `[1, x, x^2, x^3]`.
    -   `X_poly` is obtained by transforming `X` using `poly.fit_transform(X)`, which fits the transformer to `X` and then transforms it.

3.  **Ridge Regression Model Fitting**:
    -   A `Ridge` regression model is instantiated with `alpha=1.0`. The `alpha` parameter controls the regularization strength, penalizing large coefficients to prevent overfitting.
    -   The `model` is then trained by calling `model.fit(X_poly, y)`, using the polynomial features `X_poly` as input and `y` as the target variable.

4.  **Prediction**:
    -   A new array `X_fit` is created using `np.linspace` to generate 100 evenly spaced points between 0 and 5. This denser set of points is used to plot a smooth regression curve.
    -   `X_fit` is then transformed into polynomial features `X_fit_poly` using the *same* `poly` transformer used during training (`poly.transform(X_fit)`).
    -   The fitted `model` makes predictions `y_fit` on `X_fit_poly` using `model.predict(X_fit_poly)`.

5.  **Visualization**:
    -   `plt.scatter(X, y, ...)` plots the original data points.
    -   `plt.plot(X_fit, y_fit, ...)` draws the predicted regression line.
    -   `plt.legend()` displays the labels for the data and the regression line.
    -   `plt.show()` renders the plot in a new window.

## License

[Placeholder for License Information, e.g., MIT, Apache 2.0, etc.]