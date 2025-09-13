```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Generate sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(40) * 0.2

# Transform features to polynomial (degree=3)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Fit linear regression on transformed features
model = LinearRegression()
model.fit(X_poly, y)

# Predict values
X_fit = np.linspace(0, 5, 100).reshape(-1, 1)
X_fit_poly = poly.transform(X_fit)
y_fit = model.predict(X_fit_poly)

# Plot results
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X_fit, y_fit, color="red", label="Polynomial Regression")
plt.legend()
plt.show()
```