import numpy as np
import matplotlib.pyplot as plt

# Define feature extraction functions
def symmetry(image):
    res = 0.0
    for i in range(8):
        for j in range(16):
            res += abs(image[i][j] - image[15 - i][j])
    return res

def average_intensity(image):
    return np.sum(image) / 256

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression training function compatible with ±1 labels
def logistic_regression(X, Y, learning_rate=0.01, max_iter=500, tolerance=1e-6):
    Y_adjusted = (Y + 1) / 2

    X_augmented = np.c_[np.ones(X.shape[0]), X]
    w = np.zeros(X_augmented.shape[1])

    for iteration in range(max_iter):
        predictions = sigmoid(X_augmented.dot(w))
        errors = Y_adjusted - predictions
        gradient = X_augmented.T.dot(errors) / X.shape[0]

        w_new = w + learning_rate * gradient

        # Check for convergence
        if np.linalg.norm(w_new - w) < tolerance:
            break

        w = w_new

    return w

def plot_decision_boundary(X, Y, w, title, ax):
    legend_labels = set()

    for i in range(len(Y)):
        if Y[i] == 1:
            label = 'Data 1'
            ax.scatter(X[i, 0], X[i, 1], color='blue', marker='o', label=label if label not in legend_labels else "")
            legend_labels.add(label)
        else:
            label = 'Data 5'
            ax.scatter(X[i, 0], X[i, 1], color='red', marker='x', label=label if label not in legend_labels else "")
            legend_labels.add(label)

    x_vals = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 100)
    if w[2] != 0:
        y_vals = -(w[0] + w[1] * x_vals) / w[2]
        ax.plot(x_vals, y_vals, color='green', linestyle='--', linewidth=2, label="Decision Boundary")

    ax.set_xlim(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
    ax.set_ylim(X[:, 1].min() - 10, X[:, 1].max() + 10)
    ax.set_xlabel("Average Intensity")
    ax.set_ylabel("Symmetry")
    ax.set_title(title)

    ax.legend()

# Misclassification error function
def E(X, Y, w):
    predictions = np.sign(X.dot(w))
    misclassified_indices = np.where(predictions != Y)[0]
    return len(misclassified_indices) / len(Y)

# Main function
def main():
    trainFile = "FilteredZipDigits.train.txt"
    testFile = "FilteredZipDigits.test.txt"

    # Load training and testing data
    trainData = np.loadtxt(trainFile)
    testData = np.loadtxt(testFile)

    # Prepare data for logistic regression
    X_train, Y_train = [], []
    for row in trainData:
        num = int(row[0])
        if num in {1, 5}:
            img = row[1:].reshape(16, 16)
            X_train.append([average_intensity(img), symmetry(img)])
            Y_train.append(1 if num == 1 else -1)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # Train the logistic regression model and obtain the weight vector
    w = logistic_regression(X_train, Y_train)
    print("Logistic Regression Weights:", w)

    X_train_augmented = np.c_[np.ones(X_train.shape[0]), X_train]
    Ein = E(X_train_augmented, Y_train, w)
    print(f'Ein: {Ein}')

    # Process test data
    X_test, Y_test = [], []
    for row in testData:
        num = int(row[0])
        if num in {1, 5}:
            img = row[1:].reshape(16, 16)
            X_test.append([average_intensity(img), symmetry(img)])
            Y_test.append(1 if num == 1 else -1)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_test_augmented = np.c_[np.ones(X_test.shape[0]), X_test]

    # Calculate out-of-sample error Eout
    Eout = E(X_test_augmented, Y_test, w)
    print(f'Eout: {Eout}')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_decision_boundary(X_train, Y_train, w, "Training Data with Decision Boundary", ax)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_decision_boundary(X_test, Y_test, w, "Testing Data with Decision Boundary", ax)

    plt.show()

if __name__ == "__main__":
    main()
