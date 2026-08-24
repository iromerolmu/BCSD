import torch
from torch.utils.data import Dataset, DataLoader
from train_data_preprocess import binary_codes, labels
from pair_batch_sampler import PairBatchSampler

# Wrap data into a PyTorch Dataset container
class CodeSimilarityDataset(Dataset):
    def __init__(self, binary_codes, labels):
        self.binary_codes = binary_codes
        self.labels = labels

    def __len__(self):
        return len(self.binary_codes)

    def __getitem__(self, idx):
        return self.binary_codes[idx], self.labels[idx]

# Instantiate dataset wrapper
my_dataset = CodeSimilarityDataset(binary_codes, labels)

# Create the custom batch sampler using dataset's labels
custom_batch_sampler = PairBatchSampler(labels=my_dataset.labels, batch_size=32)

# Pass batch_sampler
dataloader = DataLoader(
    my_dataset, 
    batch_sampler=custom_batch_sampler
)
