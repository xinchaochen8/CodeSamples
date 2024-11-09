import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from scipy.ndimage import gaussian_filter

# Calculate the symmetry of an image
def symmetry(image):
    res = 0.0
    for i in range(8):
        for j in range(16):
            res += abs(image[i][j] - image[15 - i][j])
    return res

# Calculate the average intensity of an image
def average_intensity(image):
    return np.sum(image) / 256

# Return polynomial features up to the third degree from input matrix X
def polynomial_features(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    return np.c_[np.ones(X.shape[0]), x1, x2, x1**2, x2**2, x1*x2, x1**3, x2**3, x1**2 * x2, x1 * x2**2]

# Apply the sigmoid function with clipping to avoid overflow
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Scale features using standardization
def scale_features(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_scaled = (X_train - mean) / (std + 1e-8)
    X_test_scaled = (X_test - mean) / (std + 1e-8)
    return X_train_scaled, X_test_scaled

def logistic_regression(X, y, learning_rate=0.01, num_iter=500, verbose=True):
    N, d = X.shape
    w = np.zeros(d)
    y_adj = (y + 1) / 2

    for i in range(num_iter):
        indices = np.random.permutation(N)

        for idx in indices:
            prediction = sigmoid(X[idx].dot(w))
            gradient = (prediction - y_adj[idx]) * X[idx]
            w -= learning_rate * gradient

    return w

# Plot the decision boundary
def plot_decision_boundary(X, y, w, title, ax):
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', label='Digit 1', alpha=0.6)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], color='red', marker='x', label='Digit 5', alpha=0.6)
    
    margin = 0.5
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_poly = polynomial_features(grid)
    
    Z = sigmoid(grid_poly.dot(w))
    
    Z = Z.reshape(xx.shape)
    Z = gaussian_filter(Z, sigma=1)
    
    ax.contour(xx, yy, Z, levels=[0.5], colors='green', linestyles='-', linewidths=2)
    ax.set_xlabel("Average Intensity")
    ax.set_ylabel("Symmetry")
    ax.set_title(title)
    ax.legend()

def calculate_error(X, y, w):
    predictions = sigmoid(X.dot(w))
    predicted_labels = (predictions >= 0.5) * 2 - 1
    return np.mean(predicted_labels != y)

def main():
    # Load data
    trainFile = "FilteredZipDigits.train.txt"
    testFile = "FilteredZipDigits.test.txt"
    
    trainData = np.loadtxt(trainFile)
    testData = np.loadtxt(testFile)
    
    # Extract features and labels
    def prepare_data(data):
        X, y = [], []
        for row in data:
            num = int(row[0])
            if num in {1, 5}:
                img = row[1:].reshape(16, 16)
                X.append([average_intensity(img), symmetry(img)])
                y.append(1 if num == 1 else -1)
        return np.array(X), np.array(y)
    
    X_train, y_train = prepare_data(trainData)
    X_test, y_test = prepare_data(testData)
    
    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    X_train_poly = polynomial_features(X_train_scaled)
    X_test_poly = polynomial_features(X_test_scaled)
    
    w = logistic_regression(X_train_poly, y_train)
    
    print("Shape of X_train_poly:", X_train_poly.shape)
    print("Shape of weights w:", w.shape)
    
    train_error = calculate_error(X_train_poly, y_train, w)
    test_error = calculate_error(X_test_poly, y_test, w)
    
    print(f'Training Error: {train_error}')
    print(f'Test Error: {test_error}')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plot_decision_boundary(X_train_scaled, y_train, w, "Training Data", ax1)
    plot_decision_boundary(X_test_scaled, y_test, w, "Testing Data", ax2)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
