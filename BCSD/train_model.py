from transformers import AutoTokenizer, AutoModel
import torch.optim as optim
import torch
from InfoNCE import InfoNCELoss
from dataloader import CodeSimilarityDataset, dataloader
from evaluate_full_dataset import evaluate_test_loss
from test_dataloader import test_dataloader

# Load pre-trained CodeBERT
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModel.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)

# Initialize your loss module and optimizer
criterion = InfoNCELoss(temperature=0.07).to(device)
optimizer = optim.AdamW(encoder.parameters(), lr=2e-5)

encoder.train()
print("Starting Supervised Contrastive Training Loop...")

for epoch in range(20):  # Running for 20 epochs
    epoch_loss = 0.0

    for batch_idx, (binary_codes, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        # Tokenize text strings in the batch
        inputs = tokenizer(
            list(binary_codes),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        # Get embeddings from CodeBERT
        outputs = encoder(**inputs)

        # Use mean pooling over the sequence tokens to get one vector per function
        embeddings = outputs.last_hidden_state.mean(dim=1)

        # 3. Format Labels for the Loss Function
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

        # 4. Compute Loss and Backpropagate
        loss = criterion(embeddings, labels_tensor)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Test Pass (Across ALL test samples simultaneously)
    test_loss = evaluate_test_loss(encoder, tokenizer, test_dataloader, criterion, device)

    avg_train_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/3 | Train Loss: {avg_train_loss:.4f} | Full Test Loss: {test_loss:.4f}")
