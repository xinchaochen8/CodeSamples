import numpy as np
import matplotlib.pyplot as plt

# Define symmetry and intensity functions as before
def symmetry(image):
    res = 0.0
    for i in range(8):
        for j in range(16):
            res += abs(image[i][j] - image[15 - i][j])
    return res

# Calculate the average intensity of an image
def average_intensity(image):
    return np.sum(image) / 256

# Expands the input features to a third-order polynomial feature set.
def third_order_transform(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    return np.c_[np.ones(X.shape[0]), x1, x2, x1**2, x2**2, x1*x2, x1**3, x2**3, x1**2 * x2, x1 * x2**2]

# Perform linear regression to find initial weights
def linear_regression(X, Y):
    X_augmented = third_order_transform(X)
    w = np.linalg.pinv(X_augmented.T.dot(X_augmented)).dot(X_augmented.T).dot(Y)
    return w

# Calculate the proportion of misclassified points
def pocket(X, Y, w):
    misclassified_indices = np.where(np.sign(X.dot(w)) != Y)[0]
    return len(misclassified_indices) / len(Y)

# Pocket algorithm to improve weights
def pocket_algorithm(X, Y, w_init, maxIter=500):
    X_augmented = third_order_transform(X)
    w = w_init.copy()
    best_w = w.copy()
    min_Ein = pocket(X_augmented, Y, w)

    for iteration in range(maxIter):
        misclassified = np.where(np.sign(X_augmented.dot(w)) != Y)[0]
        if len(misclassified) == 0:
            break
        
        for i in misclassified:
            w_new = w + Y[i] * X_augmented[i]
            Ein_new = pocket(X_augmented, Y, w_new)
            if Ein_new < min_Ein:
                min_Ein = Ein_new
                best_w = w_new.copy()
            w = w_new
    
    return best_w

# Plotting function
def plot_decision_boundary(X, Y, w, title, ax):
    legend_labels = set()
    
    for i in range(len(Y)):
        label = 'Digit 1' if Y[i] == 1 else 'Digit 5'
        color, marker = ('blue', 'o') if Y[i] == 1 else ('red', 'x')
        ax.scatter(X[i, 0], X[i, 1], color=color, marker=marker, label=label if label not in legend_labels else "")
        legend_labels.add(label)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_transformed = third_order_transform(grid)

    Z = grid_transformed.dot(w)
    Z = Z.reshape(xx.shape)

    ax.contour(xx, yy, Z, levels=[0], colors='green', linestyles='-', linewidths=2)
    ax.set_xlabel("Average Intensity")
    ax.set_ylabel("Symmetry (horizontal)")
    ax.set_title(title)
    ax.legend()

def main():
    trainFile = "FilteredZipDigits.train.txt"
    testFile = "FilteredZipDigits.test.txt"
    
    trainData = np.loadtxt(trainFile)
    testData = np.loadtxt(testFile)

    X_train, Y_train = [], []
    for row in trainData:
        num = int(row[0])
        if num in [1, 5]:
            img = row[1:].reshape(16, 16)
            X_train.append([average_intensity(img), symmetry(img)])
            Y_train.append(1 if num == 1 else -1)
    
    X_train, Y_train = np.array(X_train), np.array(Y_train)

    # Perform Linear Regression and Pocket Algorithm
    w_init = linear_regression(X_train, Y_train)
    print("Initial weights from Linear Regression:", w_init)
    w_pocket = pocket_algorithm(X_train, Y_train, w_init)
    print("Improved weights after Pocket Algorithm:", w_pocket)

    # Plot training and testing data with decision boundary
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    plot_decision_boundary(X_train, Y_train, w_pocket, "Training Data with Decision Boundary", ax[0])
    
    # Prepare and plot test data
    X_test, Y_test = [], []
    for row in testData:
        num = int(row[0])
        if num in [1, 5]:
            img = row[1:].reshape(16, 16)
            X_test.append([average_intensity(img), symmetry(img)])
            Y_test.append(1 if num == 1 else -1)
    
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    plot_decision_boundary(X_test, Y_test, w_pocket, "Testing Data with Decision Boundary", ax[1])

    Ein = pocket(third_order_transform(X_train), Y_train, w_pocket)
    print(f'Ein: {Ein}')
    Eout = pocket(third_order_transform(X_test), Y_test, w_pocket)
    print(f'Eout: {Eout}')
    
    plt.show()

if __name__ == "__main__":
    main()
