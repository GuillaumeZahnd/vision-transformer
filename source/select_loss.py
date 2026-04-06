from omegaconf import DictConfig
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    """Dice Loss for binary segmentation tasks."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice loss between logits and binary targets.

        Args:
            logits: Raw (pre-sigmoid) predictions of shape (N, *).
            targets: Binary ground truth masks of shape (N, *).

        Returns:
            Scalar Dice loss averaged over the batch.
        """
        # Apply sigmoid to convert logits to probabilities [0, 1]
        probabilities = torch.sigmoid(logits)

        # Flatten tensors to (Batch, -1)
        probabilities = probabilities.view(probabilities.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)

        intersection = (probabilities * targets).sum(1)
        union = probabilities.sum(1) + targets.sum(1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_score.mean()


class Loss(Enum):
    CROSS_ENTROPY = "cross_entropy"
    DICE_BINARY = "dice_binary"


def select_loss(cfg: DictConfig) -> torch.nn:
    loss_nickname = cfg.training.loss

    if loss_nickname == Loss.CROSS_ENTROPY.value:
        loss = torch.nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)

    elif loss_nickname == Loss.DICE_BINARY.value:
        loss = BinaryDiceLoss()

    else:
        raise ValueError(f"Unknown loss '{loss_nickname}'. Valid values are {[t.value for t in Loss]}.")

    return loss
