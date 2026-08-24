import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedPairedInfoNCELoss(nn.Module):
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, 
        embeds_a: torch.Tensor, 
        embeds_b: torch.Tensor, 
        labels_a: torch.Tensor
    ) -> torch.Tensor:
    
        print(labels_a)
        device = embeds_a.device

        # Normalize embeddings along feature dimension
        embeds_a = F.normalize(embeds_a, p=2, dim=1)
        embeds_b = F.normalize(embeds_b, p=2, dim=1)

        # Similarity Matrix: Shape (B, B)
        # Entry (i, j) is the similarity between embeds_a[i] and embeds_b[j]
        sim_matrix = torch.matmul(embeds_a, embeds_b.T) / self.temperature

        # For numerical stability: subtract row-wise max
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Create Supervised Positive Mask: Shape (B, B)
        # Mask[i, j] = 1.0 if labels_a[i] == labels_b[j], else 0.0
        labels_a = labels_a.view(-1, 1)
        pos_mask = torch.eq(labels_a, labels_a.T).float().to(device)

        # Compute Log-Probabilities across the rows (Softmax denominator over all B items)
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)

        # Average log-likelihood over all positive targets in each row
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-6)

        # Loss = Negative log likelihood
        loss = -mean_log_prob_pos

        # Only average across rows that had at least one positive match in the batch
        valid_rows = pos_mask.sum(dim=1) > 0
        if valid_rows.any():
            return loss[valid_rows].mean()
        
        return loss.mean()
