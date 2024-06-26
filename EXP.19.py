import pandas as pd

def find_s_algorithm(data):
    # Initialize the most specific hypothesis
    specific_hypothesis = None
    
    for index, row in data.iterrows():
        # Get the features and label
        instance = row[:-1]
        label = row[-1]
        
        if label == 'Yes':  # Process positive examples only
            if specific_hypothesis is None:
                # Initialize the specific hypothesis with the first positive example
                specific_hypothesis = list(instance)
            else:
                # Generalize the specific hypothesis
                for i in range(len(specific_hypothesis)):
                    if specific_hypothesis[i] != instance[i]:
                        specific_hypothesis[i] = '?'
    
    return specific_hypothesis

# Example dataset
data = {
    'Example': [1, 2, 3, 4],
    'Citations': ['Some', 'Many', 'Many', 'Many'],
    'Size': ['Small', 'Big', 'Medium', 'Small'],
    'In Library': ['No', 'No', 'No', 'No'],
    'Price': ['Affordable', 'Expensive', 'Expensive', 'Affordable'],
    'Editions': ['Few', 'Many', 'Few', 'Many'],
    'Buy': ['No', 'Yes', 'Yes', 'Yes']
}

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Apply the Find-S algorithm
specific_hypothesis = find_s_algorithm(df.drop(columns=['Example']))

# Output the most specific hypothesis
print("The most specific hypothesis is:", specific_hypothesis)
