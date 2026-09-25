import torch
import torch.nn as nn
import torch.nn.functional as F

class MacroF05FocalContrastiveLoss(nn.Module):
    """
    Asymmetric Focal Contrastive Loss optimized specifically for Macro F0.5 metric.
    Penalizes False Positives (lambda_precision = 2.0) twice as heavily as False Negatives.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, lambda_precision: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_precision = lambda_precision

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply extra precision weight penalty to false positives
        precision_weights = torch.where(targets == 0, self.lambda_precision, 1.0)
        
        loss = focal_weight * bce_loss * precision_weights
        return loss.mean()