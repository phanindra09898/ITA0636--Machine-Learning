import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming we have a dataset in a CSV file named 'car_data.csv'
# Jack should replace 'car_data.csv' with the actual path to his dataset.
df = pd.read_csv('car_data.csv')

# a) Read the dataset
print("Dataset loaded successfully")

# b) Print the first five rows
print("First five rows of the dataset:")
print(df.head())

# c) Basic statistical computations
print("\nBasic statistical computations:")
print(df.describe())

# d) Display the columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)

# e) Detect and handle null values
print("\nChecking for null values:")
print(df.isnull().sum())

# Replace null values with mode value for categorical columns and median for numerical columns
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].median(), inplace=True)

print("\nDataset after handling null values:")
print(df.isnull().sum())

# f) Explore the dataset using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Feature Correlations')
plt.show()

# g) Split the data into training and test sets
# Assume the target variable is 'Price'
X = df.drop(columns=['Price'])
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical variables
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# h) Fit the model using Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# i) Predict using the model
y_pred = model.predict(X_test)

# j) Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
