import torch

def create_paired_collate_fn(tokenizer, max_length: int = 512):
    
    def collate_fn(batch):
        # Extract raw string lists and integer labels from dataset batch
        source_texts = [item["source_code"] for item in batch]
        binary_texts = [item["binary_code"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        # Dynamic batch tokenization for Source Code
        source_inputs = tokenizer(
            source_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # Dynamic batch tokenization for Binary Assembly
        binary_inputs = tokenizer(
            binary_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # Return formatted dictionary matching expected keys in training loop
        return {
            "source_inputs": source_inputs,
            "binary_inputs": binary_inputs,
            "labels": labels
        }

    return collate_fn
