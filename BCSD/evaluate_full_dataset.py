import torch

# Computes InfoNCE loss across the entire test set at once
@torch.no_grad()
def evaluate_test_loss(model, tokenizer, test_loader, criterion, device):
    model.eval()
    
    for all_binary_codes, all_labels in test_loader:
        inputs = tokenizer(
            list(all_binary_codes),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long, device=device)

        test_loss = criterion(embeddings, labels_tensor)

    model.train()
    return test_loss.item()
