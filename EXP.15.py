import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the breast cancer dataset from scikit-learn
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# a) Print the first five rows
print("First five rows of the dataset:")
print(df.head())

# b) Basic statistical computations
print("\nBasic statistical computations:")
print(df.describe())

# c) Print the columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)

# d) Detect and handle null values
print("\nChecking for null values:")
print(df.isnull().sum())

# No null values detected in scikit-learn's breast cancer dataset

# e) Split the data into training and test sets
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# f) Train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# g) Evaluate the performance of the model
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
