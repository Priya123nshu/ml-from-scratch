```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
X, y = make_classification(n_samples=300, n_features=2, n_classes=2,
                           n_informative=2, n_redundant=0, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Support Vector Classifier model
model = SVC(kernel="rbf", C=1.0, gamma="scale")
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot the predictions
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap="coolwarm", edgecolors="k")
plt.title("SVC Predictions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```