```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Features and labels
X = np.arange(10).reshape(-1, 1)  # X values: [, [1], ... ]
y = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1])  # Binary labels

# Train the model
model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
model.fit(X, y)

# Predict and evaluate
y_pred = model.predict(X)
print('Predictions:', y_pred)
print('Confusion matrix:\n', confusion_matrix(y, y_pred))
print('Classification report:\n', classification_report(y, y_pred))
```