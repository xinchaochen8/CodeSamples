import numpy as np
import matplotlib.pyplot as plt

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

# Perform linear regression to find initial weights
def linear_regression(X, Y):
    X_augmented = np.c_[np.ones(X.shape[0]), X]
    w = np.linalg.pinv(X_augmented.T.dot(X_augmented)).dot(X_augmented.T).dot(Y)
    return w

# Calculate the proportion of misclassified points
def pocket(X, Y, w):
    misclassified_indices = np.where(np.sign(X.dot(w)) != Y)[0]
    return len(misclassified_indices) / len(Y)

# Pocket algorithm to improve weights
def pocket_algorithm(X, Y, w_init, maxIter=500):
    X_augmented = np.c_[np.ones(X.shape[0]), X]
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

# Plot the decision boundary
def plot_decision_boundary(X, Y, w, title, ax):
    # Create a set to track added labels for the legend
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
    
    # Set the limits for x and y axes
    ax.set_xlim(-1, 0.2)
    ax.set_ylim(0, 140)

    # Plot the decision boundary
    ax.plot(x_vals, y_vals, color='green', linestyle='--', linewidth=2, label="Decision Boundary")
    ax.set_xlabel("Average Intensity")
    ax.set_ylabel("Symmetry (horizontal)")
    ax.set_title(title)
    ax.legend()

def main():
    trainFile = "FilteredZipDigits.train.txt"
    testFile = "FilteredZipDigits.test.txt"
    
    # Load training and testing data
    trainData = np.loadtxt(trainFile)
    testData = np.loadtxt(testFile)
        
    # Prepare data for
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
    
    # Step 1: Perform Linear Regression for initial weights
    w_init = linear_regression(X_train, Y_train)
    print("Initial weights from Linear Regression:", w_init)

    # Step 2: Apply Pocket Algorithm for improvement
    w_pocket = pocket_algorithm(X_train, Y_train, w_init)
    print("Improved weights after Pocket Algorithm:", w_pocket)

    x_vals = np.linspace(-1, 1, 100)
    if w_pocket[2] != 0:  # Ensure no division by zero
        y_vals = -(w_pocket[0] + w_pocket[1] * x_vals) / w_pocket[2]
        
        plt.plot(x_vals, y_vals, color='green', linestyle='--', label="Hypothesis Line", linewidth=2)
    
    # Ein
    XTrain_augmented = np.c_[np.ones(X_train.shape[0]), X_train]
    Ein = pocket(XTrain_augmented, Y_train, w_pocket)
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
    XTest_augumented = np.c_[np.ones(XTest.shape[0]), XTest]

    Eout = pocket(XTest_augumented, YTest, w_pocket)
    print(f'Eout: {Eout}')
    
    # Plot training data with decision boundary
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_decision_boundary(X_train, Y_train, w_pocket, "Training Data with Decision Boundary", ax)
    
    # Plot testing data with decision boundary
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_decision_boundary(XTest, YTest, w_pocket, "Testing Data with Decision Boundary", ax)

    plt.show()


if __name__ == "__main__":
    main()
