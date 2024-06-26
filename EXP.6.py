import pandas as pd
import numpy as np

# Define the dataset
data = {
    'Origin': ['Japan', 'Japan', 'Japan', 'USA', 'Japan'],
    'Manufacturer': ['Honda', 'Toyota', 'Toyota', 'Chrysler', 'Honda'],
    'Color': ['Blue', 'Green', 'Blue', 'Red', 'White'],
    'Decade': ['1980', '1970', '1990', '1980', '1980'],
    'Type': ['Economy', 'Sports', 'Economy', 'Economy', 'Economy'],
    'Example Type': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Function to implement the Find-S algorithm
def find_s_algorithm(df):
    # Filter the positive examples
    positive_examples = df[df['Example Type'] == 'Positive'].drop('Example Type', axis=1).values
    
    # Initialize the most specific hypothesis
    hypothesis = positive_examples[0].copy()
    
    # Iterate through the positive examples
    for example in positive_examples:
        for i in range(len(hypothesis)):
            if hypothesis[i] != example[i]:
                hypothesis[i] = '?'
    
    return hypothesis

# Run the Find-S algorithm
hypothesis = find_s_algorithm(df)

# Output the final hypothesis
print("The most specific hypothesis is:")
print(hypothesis)
