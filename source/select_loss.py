from omegaconf import DictConfig
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(Enum):
    CROSS_ENTROPY = "cross_entropy"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    BINARY_DICE = "binary_dice"
    BINARY_CROSS_ENTROPY_AND_DICE = "binary_cross_entropy_and_dice"


class BinaryDiceLoss(nn.Module):
    """Dice Loss for binary segmentation tasks."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice loss between logits and binary targets.

        Args:
            input: Raw (pre-sigmoid) predictions of shape (N, *).
            target: Binary ground truth masks of shape (N, *).

        Returns:
            Scalar Dice loss averaged over the batch.
        """
        # Apply sigmoid to convert logits to probabilities [0, 1]
        probabilities = torch.sigmoid(input)

        # Flatten tensors to (Batch, -1)
        probabilities = probabilities.flatten(1)
        target = target.flatten(1)

        intersection = (probabilities * target).sum(1)
        union = probabilities.sum(1) + target.sum(1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_score.mean()


class CustomBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with Logits and manual Label Smoothing.
    Standard nn.BCEWithLogitsLoss does not support label_smoothing.
    """
    def __init__(self, label_smoothing: float):
        super().__init__()

        assert 0.0 <= label_smoothing < 0.5, f"label_smoothing must be in [0, 0.5), got {label_smoothing}"
        self.label_smoothing = label_smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            # Example for label_smoothing=0.1:
            # 1 (Foreground) -> 0.95
            # 0 (Background) -> 0.05
            target = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(input, target)


class BCEAndDiceLoss(nn.Module):
    def __init__(self, weight_bce: float, weight_dice: float, label_smoothing: float):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.dice = BinaryDiceLoss()
        self.bce = CustomBCEWithLogitsLoss(label_smoothing=label_smoothing)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(input, target)
        dice_loss = self.dice(input, target)
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss


def select_loss(cfg: DictConfig) -> torch.nn:
    loss_nickname = cfg.training.get("loss")

    # Cross entropy
    if loss_nickname == Loss.CROSS_ENTROPY.value:

        valid_tasks = {"image_segmentation_multiclass", "image_classification"}
        is_valid_task = cfg.dataset.get("task_type") in valid_tasks
        is_multichannel = (
            cfg.dataset.get("nb_semantic_labels", 0) > 1 or
            cfg.dataset.get("nb_classes", 0) > 1
        )
        assert is_valid_task and is_multichannel, \
            f"'{loss_nickname}' requires a multiclass task with more than one output channels."

        loss = torch.nn.CrossEntropyLoss(label_smoothing=cfg.training.get("label_smoothing"))

    # Binary cross entropy
    elif loss_nickname == Loss.BINARY_CROSS_ENTROPY.value:

        valid_tasks = {"image_segmentation_binary", "image_classification"}
        is_valid_task = cfg.dataset.get("task_type") in valid_tasks
        is_singlechannel = (
            cfg.dataset.get("nb_semantic_labels", 0) == 1 or
            cfg.dataset.get("nb_classes", 0) == 1
        )
        assert is_valid_task and is_singlechannel, \
            f"'{loss_nickname}' requires a monoclass task with exactly one output channel."

        loss = CustomBCEWithLogitsLoss(label_smoothing=cfg.training.label_smoothing)

    # Binary Dice
    elif loss_nickname == Loss.BINARY_DICE.value:

        valid_tasks = {"image_segmentation_binary", "image_segmentation_multiclass"}
        is_valid_task = cfg.dataset.get("task_type") in valid_tasks
        assert is_valid_task, \
            f"'{loss_nickname}' requires an image segmentation task."

        loss = BinaryDiceLoss()

    # Binary cross entropy and Dice
    elif loss_nickname == Loss.BINARY_CROSS_ENTROPY_AND_DICE.value:

        valid_tasks = {"image_segmentation_binary"}
        is_valid_task = cfg.dataset.get("task_type") in valid_tasks
        is_singlechannel = cfg.dataset.get("nb_semantic_labels", 0) == 1
        assert is_valid_task and is_singlechannel, \
            f"'{loss_nickname}' requires a binary segmentation task with exactly one output channel."

        weight_bce = 1.0
        weight_dice = 1.0
        loss = BCEAndDiceLoss(
            weight_bce=weight_bce, weight_dice=weight_dice, label_smoothing=cfg.training.get("label_smoothing"))

    else:
        raise ValueError(f"Unknown loss '{loss_nickname}'. Valid values are {[t.value for t in Loss]}.")

    return loss
