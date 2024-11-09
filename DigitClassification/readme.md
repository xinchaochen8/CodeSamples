# Linear Model for Handwritten Digit Classification

This repository contains Python code to implement various machine learning algorithms for classifying handwritten digits, specifically distinguishing between the digits 1 and 5. This applies a range of classification techniques, including linear regression, logistic regression, and polynomial transformations, with each method evaluated on both training and test datasets. Additionally, gradient descent methods are implemented to optimize the classification models.

## Code Structure

The code is organized as follows:

- **`3rdPoly.py`**: Implements a 3rd-order polynomial transformation for the dataset, preparing it for use in non-linear classification.
  
- **`3rdPolyLGD.py`**: Applies logistic regression with gradient descent on the dataset after a 3rd-order polynomial transformation.
  
- **`3rdPolyLSGD.py`**: Similar to `3rdPolyLGD.py`, but uses stochastic gradient descent for logistic regression after the polynomial transformation.
  
- **`filter.py`**: Loads and filters the dataset to only include the digits 1 and 5, providing a simplified dataset for binary classification tasks.
  
- **`linear.py`**: Implements linear regression as a classifier and applies the pocket algorithm for improving classification accuracy on non-separable data.
  
- **`logisticGD.py`**: Implements logistic regression for classification using gradient descent.
  
- **`logisticSGD.py`**: Implements logistic regression for classification using stochastic gradient descent.

## Usage

1. **Data Preparation**: Ensure that the handwritten digit dataset is available and formatted correctly. Use `filter.py` to preprocess and filter the data to only include the digits 1 and 5.
  
2. **Running Classification Algorithms**:
    - To use linear regression as a classifier, run `linear.py`.
    - For logistic regression using gradient descent, execute `logisticGD.py`.
    - For logistic regression using stochastic gradient descent, execute `logisticSGD.py`.
  
3. **3rd-Order Polynomial Transformation**:
    - To apply a polynomial transformation to the features, first run `3rdPoly.py`.
    - After transformation, use `3rdPolyLGD.py` or `3rdPolyLSGD.py` to perform logistic regression with gradient descent or stochastic gradient descent, respectively, on the transformed dataset.

4. **Plotting and Evaluation**:
    - Each method script includes code for plotting the decision boundary along with the training and test data.
    - The script calculates both in-sample and out-of-sample error (`E_in` and `E_test`) and allows you to evaluate the effectiveness of each classification method.
  
## Requirements

- Python 3.x
- Libraries:
  - `numpy` (for matrix operations)
  - `matplotlib` (for plotting decision boundaries)
  - `scipy` (for optimization in some algorithms)

## Results
Each method outputs a plot of the decision boundary on both training and test data, along with error calculations. The following metrics are provided:

- **`E_in`**: Error rate on the training data.
- **`E_test`**: Error rate on the test data.

These metrics help evaluate which method performs best for separating the digits, both with and without the polynomial transformation.