import pandas as pd

# Define the dataset
data = {
    'Example': [1, 2, 3, 4],
    'Shape': ['Circular', 'Circular', 'Oval', 'Oval'],
    'Size': ['Large', 'Large', 'Large', 'Large'],
    'Color': ['Light', 'Light', 'Dark', 'Light'],
    'Surface': ['Smooth', 'Irregular', 'Smooth', 'Irregular'],
    'Thickness': ['Thick', 'Thick', 'Thin', 'Thick'],
    'Target Concept': ['+', '+', '-', '+']
}

df = pd.DataFrame(data)

# Initialize the hypothesis
hypothesis = ['0', '0', '0', '0', '0']

# Implementing Find-S algorithm
for index, row in df.iterrows():
    if row['Target Concept'] == '+':
        for i in range(len(hypothesis)):
            if hypothesis[i] == '0':
                hypothesis[i] = row.iloc[i + 1]  # Update attribute value to the instance value

# Output the most specific hypothesis
print("The most specific hypothesis is:", hypothesis)

