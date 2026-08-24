import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

class InfoNCELoss(nn.Module):

    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):

        print(labels)
        device = embeddings.device

        # Normalize the embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix (B, B)
        # Measures cosine similarity between every code sample in the batch
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # For numerical stability, subtract the maximum logit per row
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Create a mask to identify samples from the SAME group
        labels = labels.contiguous().view(-1, 1)
        
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Exclude self-similarity (a sample comparing against itself)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(labels.shape[0], device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask  # Mask now only contains 1s for true distinct positives

        # Compute the log-likelihood
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Mean over positive samples per row
        # Avoid dividing by zero if a class has no other positives in the batch
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # Final Loss (negative log likelihood)
        loss = -mean_log_prob_pos

        # Only average over samples that actually had positive pairs in the batch
        valid_indices = mask.sum(1) > 0

        if valid_indices.sum() > 0:
            loss = loss[valid_indices].mean()
        else:
            loss = loss.mean()

        last_embeddings = embeddings

        return loss
