import numpy as np
import matplotlib.pyplot as plt

np.random.seed(36)
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

# Logistic regression training function using stochastic gradient descent
def logistic_regression_sgd(X, Y, learning_rate=0.01, max_iter=500):
    # Convert -1 to 0 and +1 to 1
    Y_adjusted = (Y + 1) / 2 

    # Add bias term to X
    X_augmented = np.c_[np.ones(X.shape[0]), X]
    w = np.zeros(X_augmented.shape[1]) 

    for iteration in range(max_iter):
        # Shuffle the data
        indices = np.random.permutation(X_augmented.shape[0])
        for i in indices:
            prediction = sigmoid(X_augmented[i].dot(w))
            error = Y_adjusted[i] - prediction
            w += learning_rate * error * X_augmented[i]

    return w

def plot_decision_boundary(X, Y, w, title, ax):
    legend_labels = set()

    # Plot the data points
    for i in range(len(Y)):
        if Y[i] == 1:
            label = 'Data 1'
            ax.scatter(X[i, 0], X[i, 1], color='blue', marker='o', label=label if label not in legend_labels else "")
            legend_labels.add(label)
        else:
            label = 'Data 5'
            ax.scatter(X[i, 0], X[i, 1], color='red', marker='x', label=label if label not in legend_labels else "")
            legend_labels.add(label)

    # Create grid for decision boundary
    x_vals = np.linspace(X[:, 0].min() - 10, X[:, 0].max() + 10, 100)
    y_vals = -(w[0] + w[1] * x_vals) / w[2]
    
    ax.set_xlim(-1, 0.2)
    ax.set_ylim(0, 140)

    # Plot the decision boundary
    ax.plot(x_vals, y_vals, color='green', linestyle='--', linewidth=2, label="Decision Boundary")
    ax.set_xlabel("Average Intensity")
    ax.set_ylabel("Symmetry (horizontal)")
    ax.set_title(title)
    ax.legend()

def E(X, Y, w):
    misclassified_indices = np.where(np.sign(X.dot(w)) != Y)[0]
    return len(misclassified_indices) / len(Y)

def main():
    trainFile = "FilteredZipDigits.train.txt"
    testFile = "FilteredZipDigits.test.txt"
    
    # Load training and testing data
    trainData = np.loadtxt(trainFile)
    testData = np.loadtxt(testFile)
        
    # Prepare data for logistic regression
    X_train = []
    Y_train = []
    for row in trainData:
        num = int(row[0])
        if num == 1 or num == 5:
            img = row[1:].reshape(16, 16)
            in_res = average_intensity(img)
            sym_res = symmetry(img)
            X_train.append([in_res, sym_res])
            Y_train.append(1 if num == 1 else -1)
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    # Train the logistic regression model with SGD and obtain the weight vector
    w = logistic_regression_sgd(X_train, Y_train)
    print("Logistic Regression Weights:", w)

    # Generate x-values and calculate corresponding y-values for decision boundary
    x_vals = np.linspace(-1, 1, 100)
    if w[2] != 0:  # Ensure no division by zero
        y_vals = -(w[0] + w[1] * x_vals) / w[2]
        
        # Plot the decision boundary after plotting the data
        plt.plot(x_vals, y_vals, color='green', linestyle='--', label="Decision Boundary", linewidth=2)
    
    # Ein
    XTrain_augmented = np.c_[np.ones(X_train.shape[0]), X_train]
    Ein = E(XTrain_augmented, Y_train, w)
    print(f'Ein: {Ein}')
    
    # Eout
    XTest = []
    YTest = []
    
    for row in testData:
        num = int(row[0])
        if num == 1 or num == 5:
            img = row[1:].reshape(16, 16)
            in_res = average_intensity(img)
            sym_res = symmetry(img)
            XTest.append([in_res, sym_res])
            YTest.append(1 if num == 1 else -1)  
    
    XTest = np.array(XTest)
    YTest = np.array(YTest)
    XTest_augmented = np.c_[np.ones(XTest.shape[0]), XTest]

    Eout = E(XTest_augmented, YTest, w)
    print(f'Eout: {Eout}')

    # Plot training data with decision boundary
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_decision_boundary(X_train, Y_train, w, "Training Data with Decision Boundary", ax)
    
    # Plot testing data with decision boundary
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_decision_boundary(XTest, YTest, w, "Testing Data with Decision Boundary", ax)
    
    plt.show()


if __name__ == "__main__":
    main()
