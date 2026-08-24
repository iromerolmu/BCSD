import torch

# Computes InfoNCE loss across the entire test set at once
@torch.no_grad()
def evaluate_test_loss(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in test_loader:
        source_inputs = {k: v.to(device) for k, v in batch["source_inputs"].items()}
        binary_inputs = {k: v.to(device) for k, v in batch["binary_inputs"].items()}
        labels = batch["labels"].to(device)

        outputs_a = model(**source_inputs)
        outputs_b = model(**binary_inputs)

        embeds_a = outputs_a.last_hidden_state.mean(dim=1)
        embeds_b = outputs_b.last_hidden_state.mean(dim=1)

        loss_a2b = criterion(embeds_a, embeds_b, labels)
        loss_b2a = criterion(embeds_b, embeds_a, labels)
        batch_loss = (loss_a2b + loss_b2a) / 2.0

        total_loss += batch_loss.item()
        num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0
