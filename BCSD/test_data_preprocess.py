import json
import re

modified_data = []

# Open the file and read the raw contents completely
with open('test_binary.jsonl', 'r', encoding='utf-8') as f:
    raw_content = f.read()

# Replace all occurrences of "}{" with "}\n{" to separate objects by structure
cleaned_content = re.sub(r'\}\s*\{', '}\n{', raw_content)

# Process line by line
for line_num, line in enumerate(cleaned_content.splitlines(), 1):
    line = line.strip()
    if not line:
        continue
    try:
        modified_data.append(json.loads(line))
    except json.JSONDecodeError as e:
        print(f"Skipping a corrupted block around line {line_num}: {e}")

# 1. Extract unique names to build the reference mapping IDs
unique_functions = sorted(list(set(item['id'] for item in modified_data)))

# 2. Create the group mapping dictionary (Name -> Integer ID)
group_mapping = {func_name: idx for idx, func_name in enumerate(unique_functions)}

# 3. Process the dataset into parallel text and label arrays (duplicates are preserved)
binary_codes = []
labels = []

for item in modified_data:
    binary_codes.append(item['asm'])
    # Maps every instance to its correct ID, matching the item order exactly
    labels.append(group_mapping[item['id']])
