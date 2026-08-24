import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from test_data_preprocess import source_codes, binary_codes, labels
from pair_batch_sampler import PairBatchSampler
from collate_fn import create_paired_collate_fn

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

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Instantiate the test dataset
test_dataset = PairedSourceBinaryDataset(
    source_codes=source_codes,
    binary_codes=binary_codes,
    labels=labels
)

# Create a SINGLE DataLoader with batch_size = len(test_dataset)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=len(test_dataset),
    shuffle=False,
    collate_fn=create_paired_collate_fn(tokenizer, max_length=512)
)
