# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Preprocess the Data (not much preprocessing needed for the Iris dataset)
# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the Models
# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Logistic Regression
logistic_regression = LogisticRegression(max_iter=200)
logistic_regression.fit(X_train, y_train)

# Step 5: Evaluate the Models
# Predicting and evaluating KNN
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Predicting and evaluating Decision Tree
y_pred_tree = decision_tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)

# Predicting and evaluating Logistic Regression
y_pred_logistic = logistic_regression.predict(X_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)

# Print the results
print("Accuracy of K-Nearest Neighbors:", accuracy_knn)
print("Accuracy of Decision Tree:", accuracy_tree)
print("Accuracy of Logistic Regression:", accuracy_logistic)
