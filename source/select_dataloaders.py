from omegaconf import DictConfig
from enum import Enum
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100

from source.get_dataloaders import get_dataloaders
from source.dataset_chest_x_ray import get_chest_x_ray_datasets


class Datasets(Enum):
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    CHESTXRAY = "CHESTXRAY"


def select_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_nickname = cfg.dataset.name

    assert cfg.dataset.training_validation_split > 0 and cfg.dataset.training_validation_split <= 1, \
        "Training-validation split must be comprised in ]0, 1]."

    def _split_validation_from_training(
        dataset_training: Dataset, training_validation_split: float) -> tuple[Dataset, Dataset]:

        nb_training_samples = len(dataset_training)
        training_split = int(training_validation_split * nb_training_samples)
        validation_split = nb_training_samples - training_split
        dataset_training, dataset_validation = random_split(dataset_training, [training_split, validation_split])
        return dataset_training, dataset_validation

    # CIFAR 10
    if dataset_nickname == Datasets.CIFAR10.value:
        dataset_training = CIFAR10(root="datasets", download=True, train=True, transform=ToTensor())
        dataset_test = CIFAR10(root="datasets", download=True, train=False, transform=ToTensor())

        dataset_training, dataset_validation = _split_validation_from_training(
            dataset_training=dataset_training,
            training_validation_split=cfg.dataset.training_validation_split
        )

        dataloader_training, dataloader_validation, dataloader_test = get_dataloaders(
            cfg=cfg, dataset_training=dataset_training, dataset_validation=dataset_validation, dataset_test=dataset_test
        )

    # CIFAR 100
    elif dataset_nickname == Datasets.CIFAR100.value:
        dataset_training = CIFAR100(root="datasets", download=True, train=True, transform=ToTensor())
        dataset_test = CIFAR100(root="datasets", download=True, train=False, transform=ToTensor())

        dataset_training, dataset_validation = _split_validation_from_training(
            dataset_training=dataset_training,
            training_validation_split=cfg.dataset.training_validation_split
        )

        dataloader_training, dataloader_validation, dataloader_test = get_dataloaders(
            cfg=cfg, dataset_training=dataset_training, dataset_validation=dataset_validation, dataset_test=dataset_test
        )

    # CHEST X RAY
    elif dataset_nickname == Datasets.CHESTXRAY.value:
        dataset_training, dataset_validation, dataset_test = get_chest_x_ray_datasets(cfg=cfg)

        dataloader_training, dataloader_validation, dataloader_test = get_dataloaders(
            cfg=cfg, dataset_training=dataset_training, dataset_validation=dataset_validation, dataset_test=dataset_test
        )

    else:
        raise ValueError("Unknown dataset '{}'. Valid values are {}.".format(dataset_nickname, [e.value for e in Datasets]))

    return dataloader_training, dataloader_validation, dataloader_test
