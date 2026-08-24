import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from train_data_preprocess import source_codes, binary_codes, labels
from pair_batch_sampler import PairBatchSampler
from collate_fn import create_paired_collate_fn
import random

class PairedSourceBinaryDataset(Dataset):
    def __init__(self, source_codes, binary_codes, labels):
        assert len(source_codes) == len(binary_codes) == len(labels), \
            "All input arrays must be equal length!"
        
        self.source_codes = source_codes
        self.binary_codes = binary_codes
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.source_codes)

    def __getitem__(self, idx):
        return {
            "source_code": self.source_codes[idx],
            "binary_code": self.binary_codes[idx],
            "label": self.labels[idx]
        }

# Instantiate the new dual-stream dataset
my_dataset = PairedSourceBinaryDataset(
    source_codes=source_codes,
    binary_codes=binary_codes,
    labels=labels
)

# Instantiate custom batch sampler with dataset labels
custom_batch_sampler = PairBatchSampler(
    labels=my_dataset.labels, 
    batch_size=32
)

# Create DataLoader using batch_sampler
dataloader = DataLoader(
    my_dataset, 
    batch_sampler=custom_batch_sampler,
    collate_fn=create_paired_collate_fn(tokenizer=AutoTokenizer.from_pretrained("microsoft/codebert-base"), max_length=512)  # Optional dynamic tokenization
)
