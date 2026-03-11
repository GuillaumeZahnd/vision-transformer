from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader


def get_dataloaders(
    cfg: DictConfig,
    dataset_training: Dataset,
    dataset_validation: Dataset,
    dataset_test: Dataset
) -> tuple[DataLoader, DataLoader, DataLoader]:

    dataloader_training = DataLoader(
        dataset_training,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.environment.num_workers,
        shuffle=True,
        persistent_workers=True,
        prefetch_factor=16)

    dataloader_validation = DataLoader(
        dataset_validation,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.environment.num_workers,
        shuffle=False,
        persistent_workers=True,
        prefetch_factor=16)

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.environment.num_workers,
        shuffle=False,
        persistent_workers=True,
        prefetch_factor=16)

    return dataloader_training, dataloader_validation, dataloader_test
