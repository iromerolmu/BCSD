import torch
import random
from collections import defaultdict
from torch.utils.data import DataLoader, Sampler

class PairBatchSampler(Sampler):
    
    def __init__(self, labels, batch_size, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by label
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            # Convert label tensor to standard scalar if necessary
            if isinstance(label, torch.Tensor):
                label = label.item()
            self.label_to_indices[label].append(idx)
            
        # Filter out labels that only appear once in the dataset
        self.valid_labels = [
            lbl for lbl, idxs in self.label_to_indices.items() if len(idxs) >= 2
        ]
        
        if not self.valid_labels:
            raise ValueError("Dataset needs at least one label with >= 2 instances.")
            
        self.all_indices = list(range(len(labels)))

    def __iter__(self):
        # Shuffle overall indices and label pools per epoch
        all_indices_pool = set(self.all_indices)
        
        while len(all_indices_pool) >= self.batch_size:
            batch = []
            
            # Pick a random label and draw at least 2 indices for it
            chosen_label = random.choice(self.valid_labels)
            
            # Draw 2 indices for this label that are still available
            available_for_label = [
                idx for idx in self.label_to_indices[chosen_label] 
                if idx in all_indices_pool
            ]
            
            # pick another label if not enough indices are available for the chosen label
            if len(available_for_label) < 2:
                pair = random.sample(self.label_to_indices[chosen_label], 2)
                # Remove them if they happen to be in the pool
                all_indices_pool.difference_update(pair)
            else:
                pair = random.sample(available_for_label, 2)
                all_indices_pool.difference_update(pair)
                
            batch.extend(pair)
            
            # Fill the rest of the batch with remaining indices
            needed = self.batch_size - len(batch)
            fillers = random.sample(list(all_indices_pool), needed)
            all_indices_pool.difference_update(fillers)
            batch.extend(fillers)
            
            # Shuffle batch
            random.shuffle(batch)
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.all_indices) // self.batch_size
        else:
            return (len(self.all_indices) + self.batch_size - 1) // self.batch_size
