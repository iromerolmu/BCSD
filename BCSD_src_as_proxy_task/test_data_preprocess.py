import json
import re

source_codes = []
binary_codes = []
labels = []

# Read file and clean concatenated JSON blocks
with open('merged_output_test.jsonl', 'r', encoding='utf-8') as f:
    raw_content = f.read()

cleaned_content = re.sub(r'\}\s*\{', '}\n{', raw_content)

# Parse and flatten 1-to-N relationships
for line_num, line in enumerate(cleaned_content.splitlines(), 1):
    line = line.strip()
    if not line:
        continue
    
    try:
        data = json.loads(line)
        
        # Fallback: fill source code from ID or first binary if source_code is empty
        source = data.get("source_code", "").strip()
        if not source:
            source = data["binaries"][0]  # Uses the primary binary variant as source proxy
            
        group_label = data["label"]

        # Flatten the list of binaries into individual parallel samples
        for binary_variant in data["binaries"]:
            source_codes.append(source)
            binary_codes.append(binary_variant)
            labels.append(group_label)

    except json.JSONDecodeError as e:
        print(f"Skipping block around line {line_num}: {e}")

print(f"Loaded {len(source_codes)} individual pairs across all binary variants!")

