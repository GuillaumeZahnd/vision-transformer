from omegaconf import DictConfig
from enum import Enum
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torchvision.datasets import CIFAR100


class Datasets(Enum):
    CIFAR100 = "CIFAR100"


def select_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_nickname = cfg.dataset.name

    assert cfg.dataset.training_validation_split > 0 and cfg.dataset.training_validation_split <= 1, \
        "Training-validation split must be comprised in ]0, 1]."

    def _split_validation_from_training(
        dataset_training: Dataset, training_validation_split: float) -> tuple[Dataset, Dataset]:

        nb_training_samples = len(dataset_training)
        training_split = int(training_validation_split * nb_training_samples)
        validation_split = nb_training_samples - training_split
        dataset_training, dataset_validation = random_split(
            dataset_training, [training_split, validation_split])
        return dataset_training, dataset_validation


    if dataset_nickname == Datasets.CIFAR100.value:
        dataset_training = CIFAR100(root="datasets", download=True, train=True, transform=ToTensor())
        dataset_test = CIFAR100(root="datasets", download=True, train=False, transform=ToTensor())

        dataset_training, dataset_validation = _split_validation_from_training(
            dataset_training=dataset_training,
            training_validation_split=cfg.dataset.training_validation_split)

        dataloader_training = DataLoader(
            dataset_training, batch_size=cfg.training.batch_size, shuffle=True,
            num_workers=cfg.environment.num_workers, persistent_workers=True, prefetch_factor=16)
        dataloader_validation = DataLoader(
            dataset_validation, batch_size=cfg.training.batch_size, shuffle=False,
            num_workers=cfg.environment.num_workers, persistent_workers=True, prefetch_factor=16)
        dataloader_test = DataLoader(
            dataset_test, batch_size=cfg.training.batch_size, shuffle=False,
            num_workers=cfg.environment.num_workers, persistent_workers=True, prefetch_factor=16)

        return dataloader_training, dataloader_validation, dataloader_test

    else:
        raise ValueError("Unknown dataset '{}'. Valid values are {}.".format(dataset_nickname, [e.value for e in Datasets]))
