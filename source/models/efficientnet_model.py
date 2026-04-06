import torch
import torch.nn as nn
from enum import Enum
from omegaconf import DictConfig
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class Task(Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"

def _init_weights_kaiming_he(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class EfficientNetModel(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # 1. Load the Backbone (EfficientNet-B0)
        # We exclude the original classifier head to adapt it to our Task
        weights = EfficientNet_B0_Weights.DEFAULT if cfg.model.pretrained else None
        base_model = efficientnet_b0(weights=weights)
        
        # Modify the first layer of the backbone to support the actual channel count of the dataset
        original_conv = base_model.features[0][0]
        base_model.features[0][0] = nn.Conv2d(
            in_channels=cfg.dataset.nb_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )        
        
        # Extract features (the 'features' part of torchvision EfficientNet)
        self.backbone = base_model.features 
        
        # EfficientNet-B0 ends with 1280 channels before the pooling layer
        self.out_channels = 1280

        # 2. Task-Specific Heads
        if cfg.model.task == Task.CLASSIFICATION.value:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p=cfg.model.dropout, inplace=True),
                nn.Linear(self.out_channels, cfg.dataset.nb_classes)
            )

        elif cfg.model.task == Task.SEGMENTATION.value:
            # Simple Decoder for Segmentation
            # In a real medical setup, you'd use a Feature Pyramid or U-Net style decoder
            self.decoder = nn.Sequential(
                nn.Conv2d(self.out_channels, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, cfg.dataset.nb_semantic_labels, kernel_size=1)
            )

        if not cfg.model.pretrained:
            self.apply(_init_weights_kaiming_he)

    def forward(self, input_images: torch.Tensor):
        """
        Forward pass for EfficientNet-B0.
        """
        # Extract features: (B, 3, H, W) -> (B, 1280, H/32, W/32)
        features = self.backbone(input_images)

        if self.cfg.model.task == Task.CLASSIFICATION.value:
            logits = self.classifier(features)
            return logits

        elif self.cfg.model.task == Task.SEGMENTATION.value:
            # Map features to classes and upscale back to original resolution
            mask_logits = self.decoder(features)
            return torch.nn.functional.interpolate(
                mask_logits, 
                size=(self.cfg.dataset.image_height, self.cfg.dataset.image_width), 
                mode="bilinear", 
                align_corners=False
            )
