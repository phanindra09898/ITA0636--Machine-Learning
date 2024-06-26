import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# a) Read the house dataset using Pandas
df = pd.read.DataFrame("C:/Users/phani/OneDrive/Desktop/ML_datasets/HousePricePrediction.csv")  # Replace 'house_data.csv' with the actual path to your dataset

# b) Print the first five rows
print("First five rows of the dataset:")
print(df.head())

# c) Basic statistical computations
print("\nBasic statistical computations:")
print(df.describe())

# d) Print the columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)

# e) Detect and handle null values
print("\nChecking for null values:")
print(df.isnull().sum())

# Replace null values with mode for categorical columns and median for numerical columns
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
X = df.drop(columns=['price'])  # Features
y = df['price']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# h) Train a Linear Regression model to predict house prices
model = LinearRegression()
model.fit(X_train, y_train)

# i) Predict the price of a house (example)
# Example: Suppose we have a new house with the following features
new_house = pd.DataFrame({
    'bedrooms': [4],
    'bathrooms': [2],
    'sqft_living': [2000],
    'sqft_lot': [5000],
    'floors': [2],
    'waterfront': ['No'],  # Example categorical feature
    'view': ['None'],      # Example categorical feature
    'condition': ['Good'], # Example categorical feature
    'grade': [7],          # Example categorical feature
    'sqft_above': [2000],
    'sqft_basement': [0],
    'yr_built': [1990],
    'yr_renovated': [0],
    'zipcode': [98001],
    'lat': [47.32],
    'long': [-122.21],
    'sqft_living15': [2100],
    'sqft_lot15': [4000]
})

# Transform categorical variables in the new_house example
new_house = pd.get_dummies(new_house, drop_first=True)

# Predict the price
predicted_price = model.predict(new_house)
print("\nPredicted price of the house:", predicted_price[0])
