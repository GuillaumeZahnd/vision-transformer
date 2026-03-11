from omegaconf import DictConfig
from enum import Enum
import torch
import torchmetrics


class TaskType(Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_SEGMENTATION_BINARY = "image_segmentation_binary"


def select_accuracy(cfg: DictConfig) -> torch.nn:
    task_nickname = cfg.dataset.task_type

    if task_nickname == TaskType.IMAGE_CLASSIFICATION.value:
        accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=cfg.dataset.nb_classes
        )

    elif task_nickname == TaskType.IMAGE_SEGMENTATION_BINARY.value:
        accuracy = torchmetrics.classification.BinaryAccuracy()

    else:
        raise ValueError("Unknown task type '{}'. Valid values are {}.".format(task_nickname, [e.value for e in TaskType]))

    return accuracy
