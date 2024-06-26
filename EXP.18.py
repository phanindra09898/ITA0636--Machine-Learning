import numpy as np

def candidate_elimination(examples):
    num_attributes = len(examples[0]) - 1  # Number of attributes in the examples
    specific_hypothesis = ['0'] * num_attributes  # most specific hypothesis
    general_hypothesis = [['?'] * num_attributes]  # most general hypothesis

    for example in examples:
        if example[-1] == 'Yes':  # positive example
            for i in range(num_attributes):
                if specific_hypothesis[i] == '0':
                    specific_hypothesis[i] = example[i]
                elif specific_hypothesis[i] != example[i]:
                    specific_hypothesis[i] = '?'
                    
            # Remove inconsistent hypotheses from general_hypothesis
            general_hypothesis = [g for g in general_hypothesis if all(
                g[i] == '?' or g[i] == example[i] for i in range(num_attributes))]

        else:  # negative example
            new_general_hypothesis = []
            for g in general_hypothesis:
                for i in range(num_attributes):
                    if g[i] == '?':
                        for value in np.unique([e[i] for e in examples if e[-1] == 'Yes']):
                            if value != example[i]:
                                new_hypothesis = g[:]
                                new_hypothesis[i] = value
                                if all(specific_hypothesis[j] == '?' or specific_hypothesis[j] == new_hypothesis[j] for j in range(num_attributes)):
                                    new_general_hypothesis.append(new_hypothesis)
            general_hypothesis = new_general_hypothesis

    return specific_hypothesis, general_hypothesis

# Example dataset
dataset = [
    ['Some', 'Small', 'No', 'Affordable', 'Few', 'No', 'No'],
    ['Many', 'Big', 'No', 'Expensive', 'Many', 'Yes', 'Yes'],
    ['Many', 'Medium', 'No', 'Expensive', 'Few', 'Yes', 'Yes'],
    ['Many', 'Small', 'No', 'Affordable', 'Many', 'Yes', 'Yes']
]

# Applying the Candidate-Elimination algorithm
specific, general = candidate_elimination(dataset)
print("Specific hypothesis:", specific)
print("General hypotheses:", general)
