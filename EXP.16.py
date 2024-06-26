# Function to find the most specific hypothesis using FIND-S algorithm
def find_s_algorithm(training_data):
    # Initialize the most specific hypothesis
    hypothesis = None
    
    for example in training_data:
        # Unpack the example
        attributes, label = example[:-1], example[-1]
        
        # Only process positive examples
        if label == 'Yes':
            if hypothesis is None:
                # Initialize the hypothesis as the first positive example
                hypothesis = list(attributes)
            else:
                # Update the hypothesis
                for i in range(len(hypothesis)):
                    if hypothesis[i] != attributes[i]:
                        hypothesis[i] = '?'
    
    return hypothesis

# Example training data
training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

# Applying FIND-S algorithm on the example training data
most_specific_hypothesis = find_s_algorithm(training_data)

print("The most specific hypothesis is:", most_specific_hypothesis)
