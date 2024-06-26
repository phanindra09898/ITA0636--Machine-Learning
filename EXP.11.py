import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load the Iris dataset
iris = pd.read_csv("C:/Users/phani/OneDrive/Desktop/ML_datasets/IRIS.csv")

# Step 2: Visualize the data
plt.figure(figsize=(10, 6))
species_colors = {'Iris-setosa': 'r', 'Iris-versicolor': 'g', 'Iris-virginica': 'b'}
colors = iris['species'].map(species_colors)
plt.scatter(iris['sepal_width'], iris['sepal_length'], c=colors, s=50)
plt.title('Sepal Width vs Sepal Length')
plt.xlabel('Sepal Width')
plt.ylabel('Sepal Length')
plt.show()

# Step 3: Split the data into training and test sets
X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict with new test data
new_data = [[5, 3, 1, 0.3]]
predicted_species = model.predict(new_data)

# Output the predicted species
predicted_species[0]
