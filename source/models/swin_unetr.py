from omegaconf import DictConfig
import torch.nn as nn
from monai.networks.nets import SwinUNETR as SwinUNETR


class SwinUNETRModel(nn.Module):
    """
    Hatamizadeh et al. (2022). "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images."
    https://arxiv.org/abs/2201.01266
    """
    def __init__(self, cfg):
        super().__init__()
        self.model = SwinUNETR(
            in_channels=cfg.dataset.nb_channels,
            out_channels=cfg.dataset.nb_semantic_labels,
            feature_size=48,
            spatial_dims=2
        )


    def forward(self, input_images):
        return self.model(input_images)
