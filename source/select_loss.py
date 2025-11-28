from omegaconf import DictConfig
from enum import Enum
import torch


class Loss(Enum):
    CROSS_ENTROPY_LOSS = "cross_entropy"


def select_loss(cfg: DictConfig) -> torch.nn:
    loss_nickname = cfg.training.loss

    if loss_nickname == Loss.CROSS_ENTROPY_LOSS.value:
        return torch.nn.CrossEntropyLoss()

    else:
        raise ValueError("Unknown loss '{}'. Valid values are {}.".format(loss_nickname, [e.value for e in Loss]))
