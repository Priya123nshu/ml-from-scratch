# Support Vector Classifier (SVC)

## Overview
This Python script demonstrates the application of a Support Vector Classifier (SVC) for binary classification using the scikit-learn library. It generates a synthetic 2-feature, 2-class dataset, splits it into training and testing sets, trains an SVC model with an RBF kernel, evaluates its accuracy, and visualizes the classification predictions on the test set.

## Installation
This script requires the following Python libraries:
*   `numpy`
*   `matplotlib`
*   `scikit-learn`

You can install them using pip:
```bash
pip install numpy matplotlib scikit-learn
```

## Usage
This is a self-contained script that generates its own data, trains a model, and displays results. To run the script, save the code as a Python file (e.g., `svc_demo.py`) and execute it from your terminal:

```bash
python svc_demo.py
```

Upon execution, the script will:
1.  Generate a synthetic dataset.
2.  Train an SVC model.
3.  Print the accuracy of the model on the test set.
4.  Display a scatter plot visualizing the model's predictions on the test data.

## Inputs & Outputs

### Inputs
The script does not take any external inputs. It internally generates a synthetic dataset for demonstration purposes using `sklearn.datasets.make_classification`.

### Outputs
*   **Console Output**: The script prints the accuracy score of the classifier on the test data to the console.
    ```
    Accuracy: 0.9666666666666667
    ```
    (The exact accuracy may vary slightly based on scikit-learn versions, but `random_state` is fixed for reproducibility in this script).
*   **Graphical Output**: A `matplotlib` window will appear displaying a scatter plot titled "SVC Predictions".
    *   The plot shows the data points from the test set (`X_test`).
    *   Points are colored according to the `y_pred` (predicted class by the SVC model) using the "coolwarm" colormap.
    *   Black edge colors (`edgecolors="k"`) are added for better visibility of individual points.
    *   The x-axis is labeled "Feature 1" and the y-axis "Feature 2".

## Explanation
The script performs the following steps:

1.  **Dataset Generation**: `sklearn.datasets.make_classification` is used to create a synthetic dataset of 300 samples (`n_samples=300`), with 2 features (`n_features=2`), 2 classes (`n_classes=2`), and both features being informative (`n_informative=2`, `n_redundant=0`). A `random_state` is set for reproducibility.

2.  **Data Splitting**: The generated dataset (`X`, `y`) is split into training and testing sets using `sklearn.model_selection.train_test_split`. 70% of the data is used for training and 30% for testing (`test_size=0.3`), also with a fixed `random_state`.

3.  **Model Initialization and Training**:
    *   An `SVC` model from `sklearn.svm` is initialized.
    *   It uses a Radial Basis Function (RBF) kernel (`kernel="rbf"`).
    *   The regularization parameter `C` is set to `1.0`.
    *   The kernel coefficient `gamma` is set to `"scale"`, meaning it will be `1 / (n_features * X.var())`.
    *   The model is then trained on the training data (`X_train`, `y_train`) using `model.fit()`.

4.  **Prediction**: The trained model makes predictions on the unseen test data (`X_test`) using `model.predict()`, storing the results in `y_pred`.

5.  **Evaluation**: The accuracy of the model's predictions (`y_pred`) against the actual test labels (`y_test`) is calculated using `sklearn.metrics.accuracy_score` and printed to the console.

6.  **Visualization**:
    *   `matplotlib.pyplot.scatter` is used to create a 2D scatter plot.
    *   The x-coordinates are the first feature of `X_test`, and y-coordinates are the second feature of `X_test`.
    *   The color of each point is determined by its predicted class (`y_pred`), using the "coolwarm" colormap.
    *   The plot is given a title and axis labels, then displayed.

## License
[Add your desired license here, e.g., MIT, Apache 2.0, etc.]