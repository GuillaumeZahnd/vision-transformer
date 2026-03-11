from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class ChestXRayDataset(Dataset):

    def __init__(self, dataset_dir: str, site_dir: str) -> None:

        self.dataset_dir = dataset_dir
        self.site_dir = site_dir
        self.to_tensor = ToTensor()
        self.file_names = []

        def _assert_mask_existence(dataset_dir: str, site_dir: str, file_name: str) -> None:
            path_to_mask = Path(dataset_dir) / site_dir / "mask" / file_name
            assert path_to_mask.is_file(), f"Mask file not found: '{path_to_mask}'."

        img_dir = Path(dataset_dir) / site_dir / "img"

        if img_dir.is_dir():
            for f in img_dir.iterdir():
                if f.is_file() and f.suffix.lower() == ".png":
                    _assert_mask_existence(
                        dataset_dir=dataset_dir,
                        site_dir=site_dir,
                        file_name=f.name
                    )
                    self.file_names.append(f.name)

        assert len(self.file_names) > 0, f"Empty dataset at {Path(dataset_dir) / site_dir}."


    def __len__(self):
        return len(self.file_names)


    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        img_path = Path(self.dataset_dir) / self.site_dir / "img" / file_name
        mask_path = Path(self.dataset_dir) / self.site_dir / "mask" / file_name

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        return image, mask


def get_chest_x_ray_datasets(cfg: DictConfig) -> tuple[Dataset, Dataset, Dataset]:

    dataset_training = ChestXRayDataset(dataset_dir=cfg.dataset.dataset_dir, site_dir="Darwin")
    dataset_validation = ChestXRayDataset(dataset_dir=cfg.dataset.dataset_dir, site_dir="Montgomery")
    dataset_test = ChestXRayDataset(dataset_dir=cfg.dataset.dataset_dir, site_dir="Shenzhen")

    return dataset_training, dataset_validation, dataset_test
