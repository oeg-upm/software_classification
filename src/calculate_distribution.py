import pandas as pd
import ast

# Load the CSV into a DataFrame
df = pd.read_csv('software_classification_v2.csv')

# Convert the string representation of dictionaries into actual dictionaries
df['types_software'] = df['types_software'].apply(ast.literal_eval)

# Initialize a dictionary to hold the total counts
type_counts = {
    'Library': 0,
    'Benchmark': 0,
    'Service': 0,
    'Workflow': 0,
    'Script': 0,
    'Notebook': 0,
    'Other': 0
}

# Iterate over each row and count the occurrences of each type in the 'types_software' column
for types_dict in df['types_software']:
    for key, count in types_dict.items():
        if key in type_counts:
            type_counts[key] += count

# Display the results
print("Occurrences of each type:")
for type_name, count in type_counts.items():
    print(f"{type_name}: {count} {100*(count/107541)}")