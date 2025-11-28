from omegaconf import DictConfig
from enum import Enum
import torch
from collections.abc import Callable


class Optimizer(Enum):
    SGD = "sgd"
    ADAM = "adam"


def select_optimizer(cfg: DictConfig, parameters: Callable) -> torch.optim.Optimizer:
    optimizer_nickname = cfg.training.optimizer

    if optimizer_nickname == Optimizer.SGD.value:
        return torch.optim.SGD(params=parameters, lr=cfg.training.learning_rate)

    elif optimizer_nickname == Optimizer.ADAM.value:
        return torch.optim.Adam(params=parameters, lr=cfg.training.learning_rate)

    else:
        raise ValueError(
            "Unknown optimizer '{}'. Valid values are {}.".format(optimizer_nickname, [e.value for e in Optimizer]))
