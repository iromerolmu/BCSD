import json
from collections import defaultdict


def merge_jsonl(source_file: str, asm_file: str, output_file: str):
    merged_data = defaultdict(lambda: {"source_code": "", "binaries": []})

    # Process Source Code File
    print(f"Reading source codes from {source_file}...")
    with open(
        source_file, "r", encoding="utf-8", errors="replace"
    ) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                item_id = data.get("id")
                source = data.get("asm") or data.get("src", "")
                if item_id:
                    merged_data[item_id]["source_code"] = source
            except json.JSONDecodeError as e:
                print(
                    f"⚠️ Skipping line {line_num} in {source_file} (Invalid JSON: {e})"
                )

    # Process Assembly / Binaries File
    print(f"Reading assembly snippets from {asm_file}...")
    with open(asm_file, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                item_id = data.get("id")
                asm_code = data.get("asm") or data.get("binary", "")
                if item_id and asm_code:
                    merged_data[item_id]["binaries"].append(asm_code)
            except json.JSONDecodeError as e:
                print(
                    f"⚠️ Skipping line {line_num} in {asm_file} (Invalid JSON: {e})"
                )

    # Write Output File
    print(f"Writing merged data to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as out_f:
        for idx, (item_id, content) in enumerate(merged_data.items()):
            merged_entry = {
                "id": item_id,
                "label": idx,
                "source_code": content["source_code"],
                "binaries": content["binaries"],
            }
            out_f.write(json.dumps(merged_entry) + "\n")

    print(
        f"✅ Done! Processed {len(merged_data)} unique IDs into '{output_file}'."
    )


if __name__ == "__main__":
    merge_jsonl(
        "test_source.jsonl", "test_binary.jsonl", "merged_output_test.jsonl"
    )
