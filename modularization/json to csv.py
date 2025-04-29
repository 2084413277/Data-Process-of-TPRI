import json
import pandas as pd

# Load the JSON file
file_path = 'LW.TT.N6TS_F_FW.json'  # Modify with the correct file path if needed
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract relevant information from JSON (assuming the structure provided earlier)
tagvals = data['tagvals']

# Convert to a DataFrame
df = pd.DataFrame(tagvals)

# Save to CSV
csv_file_path = file_path.replace('.json', '.csv')
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved to: {csv_file_path}")
