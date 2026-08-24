import torch
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from InfoNCE import SupervisedPairedInfoNCELoss
from dataloader import dataloader
from test_dataloader import test_dataloader
from evaluate_full_dataset import evaluate_test_loss

# Load Pre-trained CodeBERT Model and Tokenizer
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModel.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)

# Initialize Loss and Optimizer
criterion = SupervisedPairedInfoNCELoss(temperature=0.07).to(device)
optimizer = optim.AdamW(encoder.parameters(), lr=2e-5)

encoder.train()
print("Starting Supervised Contrastive Training Loop (Source vs. Binary)...")

# Training Loop
num_epochs = 20

for epoch in range(num_epochs):
    epoch_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Extract tokenized inputs and labels from collate_fn batch
        source_inputs = {k: v.to(device) for k, v in batch["source_inputs"].items()}
        binary_inputs = {k: v.to(device) for k, v in batch["binary_inputs"].items()}
        labels = batch["labels"].to(device)

        # Get embeddings for Source Code (Stream A)
        outputs_a = encoder(**source_inputs)
        embeds_a = outputs_a.last_hidden_state.mean(dim=1)

        # Get embeddings for Binary Disassembly (Stream B)
        outputs_b = encoder(**binary_inputs)
        embeds_b = outputs_b.last_hidden_state.mean(dim=1)

        # Compute Symmetric Supervised InfoNCE Loss
        loss_a2b = criterion(embeds_a, embeds_b, labels)
        loss_b2a = criterion(embeds_b, embeds_a, labels)
        loss = (loss_a2b + loss_b2a) / 2.0

        # Backpropagation and Optimization Step
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Evaluate Test Loss at the end of each epoch
    test_loss = evaluate_test_loss(encoder, test_dataloader, criterion, device)

    avg_train_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f}")
