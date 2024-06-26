import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Hypothetical dataset creation
data = {
    'Age': [25, 30, 35, 40, 45, 50, 28, 32, 36, 29, 41, 34, 27, 38, 31],
    'Occupation': ['Engineer', 'Doctor', 'Artist', 'Engineer', 'Doctor', 'Artist', 'Engineer', 'Doctor', 'Artist', 'Engineer', 'Doctor', 'Artist', 'Engineer', 'Doctor', 'Artist'],
    'Annual Income': [50000, 80000, 40000, 60000, 100000, 45000, 52000, 82000, 48000, 58000, 110000, 46000, 53000, 88000, 49000],
    'Credit Score': [700, 750, 600, 720, 770, 610, 710, 760, 620, 730, 780, 630, 740, 790, 640]
}

# Create DataFrame
df = pd.DataFrame(data)

# a) Print the first five rows
print("First five rows of the dataset:")
print(df.head())

# b) Basic statistical computations
print("\nBasic statistical computations:")
print(df.describe())

# c) The columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)

# d) Detect and handle null values
print("\nChecking for null values:")
print(df.isnull().sum())

# Assuming we had null values, let's introduce some for demonstration
df.loc[2, 'Credit Score'] = np.nan
df.loc[10, 'Credit Score'] = np.nan
print("\nDataset with some null values introduced:")
print(df)

# Replace null values with mode value of 'Credit Score'
mode_value = df['Credit Score'].mode()[0]
df['Credit Score'].fillna(mode_value, inplace=True)
print("\nDataset after replacing null values with mode:")
print(df)

# e) Explore the dataset using box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Occupation', y='Credit Score', data=df)
plt.title('Credit Scores Based on Occupation')
plt.show()

# f) Split the data into training and testing sets
X = df[['Age', 'Occupation', 'Annual Income']]
X = pd.get_dummies(X, drop_first=True)
y = df['Credit Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# g) Fit the model using Naive Bayes Classifier
model = GaussianNB()
model.fit(X_train, y_train)

# i) Predict using the model
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
