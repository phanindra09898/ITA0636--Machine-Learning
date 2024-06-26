import pandas as pd

# Load the training examples from CSV
data = pd.DataFrame({
    'Example': [1, 2, 3, 4],
    'Shape': ['Circular', 'Circular', 'Oval', 'Oval'],
    'Size': ['Large', 'Large', 'Large', 'Large'],
    'Color': ['Light', 'Light', 'Dark', 'Light'],
    'Surface': ['Smooth', 'Irregular', 'Smooth', 'Irregular'],
    'Thickness': ['Thick', 'Thick', 'Thin', 'Thick'],
    'Target Concept': ['+', '+', '-', '+']
})

# Initialize G and S
attributes = ['Shape', 'Size', 'Color', 'Surface', 'Thickness']
G = [{'Shape': '?', 'Size': '?', 'Color': '?', 'Surface': '?', 'Thickness': '?'}]
S = [{'Shape': 'None', 'Size': 'None', 'Color': 'None', 'Surface': 'None', 'Thickness': 'None'}]

# Process each training example
for index, row in data.iterrows():
    target = row['Target Concept']
    attributes_values = row.drop(['Example', 'Target Concept']).to_dict()

    if target == '+':  # Positive example
        # Remove from G any hypothesis that is inconsistent with attributes
        G = [g for g in G if all(g[attr] == '?' or g[attr] == attributes_values[attr] for attr in g)]

        # For each hypothesis in S that is not consistent with attributes, generalize it
        for s in S.copy():
            if any(s[attr] != '?' and s[attr] != attributes_values[attr] for attr in s):
                for attr in s:
                    if s[attr] == 'None':
                        s[attr] = attributes_values[attr]
                    elif s[attr] != '?' and s[attr] != attributes_values[attr]:
                        s[attr] = '?'
        
    elif target == '-':  # Negative example
        # Remove from S any hypothesis that is inconsistent with attributes
        S = [s for s in S if any(s[attr] == '?' or s[attr] == attributes_values[attr] for attr in s)]

        # For each hypothesis in G that is not consistent with attributes, specialize it
        for g in G.copy():
            if all(g[attr] == '?' or g[attr] == attributes_values[attr] for attr in g):
                for attr in g:
                    if g[attr] == '?':
                        for value in set(data[attr].unique()) - {attributes_values[attr]}:
                            new_hypothesis = g.copy()
                            new_hypothesis[attr] = value
                            if new_hypothesis not in G:
                                G.append(new_hypothesis)
                G.remove(g)

# Output the final version of G and S
print("Final version of G:")
for hypothesis in G:
    print(hypothesis)

print("\nFinal version of S:")
for hypothesis in S:
    print(hypothesis)
