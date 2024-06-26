import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Example usage
if __name__ == "__main__":
    # Generate some example data
    np.random.seed(42)
    X_train = np.random.rand(100, 2)
    y_train = np.random.randint(0, 2, 100)

    X_test = np.random.rand(10, 2)

    # Fit KNN model
    knn = KNN(k=3)
    knn.fit(X_train, y_train)

    # Predict on new data
    predictions = knn.predict(X_test)

    # Output predictions
    print("Predictions:", predictions)

    # Plot the results
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='Class 1')
    plt.scatter(X_test[:, 0], X_test[:, 1], color='green', marker='x', label='Test Points')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("K-Nearest Neighbors")
    plt.legend()
    plt.show()
