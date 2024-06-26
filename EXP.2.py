import pandas as pd

# Define the dataset
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'Air Temp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'Enjoy Sport': ['Yes', 'Yes', 'No', 'Yes']
}

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data)

# Function to implement the Find-S algorithm
def find_s(data):
    # Initialize the hypothesis with the most specific hypothesis
    hypothesis = ['0'] * (len(data.columns) - 1)
    
    # Iterate over the examples
    for i in range(len(data)):
        # Check if the example is positive
        if data.iloc[i, -1] == 'Yes':
            # Update the hypothesis
            for j in range(len(hypothesis)):
                if hypothesis[j] == '0':
                    hypothesis[j] = data.iloc[i, j]
                elif hypothesis[j] != data.iloc[i, j]:
                    hypothesis[j] = '?'
    return hypothesis

# Apply the Find-S algorithm
most_specific_hypothesis = find_s(df)

# Print the result
print(f"The most specific hypothesis is: {most_specific_hypothesis}")
