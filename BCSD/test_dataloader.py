import torch
from torch.utils.data import Dataset, DataLoader
from test_data_preprocess import binary_codes, labels  # Load test data here

class CodeSimilarityDataset(Dataset):
    def __init__(self, binary_codes, labels):
        self.binary_codes = binary_codes
        self.labels = labels

    def __len__(self):
        return len(self.binary_codes)

    def __getitem__(self, idx):
        return self.binary_codes[idx], self.labels[idx]

# Instantiate dataset with test data
test_dataset = CodeSimilarityDataset(binary_codes, labels)

# Set batch_size equal to the entire dataset length
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=len(test_dataset),
    shuffle=False                 
)

# Fetch all test data in a single iteration
for all_binary_codes, all_labels in test_dataloader:
    print(f"Loaded ALL {len(all_binary_codes)} test samples simultaneously!")
