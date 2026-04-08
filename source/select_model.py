from omegaconf import DictConfig
from enum import Enum
import torch


from source.models.vision_transformer_model import VisionTransformerModel
from source.models.efficientnet_model import EfficientNetModel
from source.models.hybrid_trans_unet_pyramid import HybridTransUNet
from source.models.swin_unetr import SwinUNETRModel


class ModelType(Enum):
    CUSTOM_VIT = "custom_vit"
    EFFICIENTNET = "efficientnet"
    HYBRID = "hybrid"
    SWIN_UNETR = "swin_unetr"


def select_model(cfg: DictConfig) -> torch.nn:
    model_nickname = cfg.model.model_type

    if model_nickname == ModelType.CUSTOM_VIT.value:
        return VisionTransformerModel(cfg=cfg)

    elif model_nickname == ModelType.EFFICIENTNET.value:
        return EfficientNetModel(cfg=cfg)

    elif model_nickname == ModelType.HYBRID.value:
        return HybridTransUNet(cfg=cfg)

    elif model_nickname == ModelType.SWIN_UNETR.value:
        valid_tasks = {"image_segmentation_binary", "image_segmentation_multiclass"}
        is_valid_task = cfg.dataset.get("task_type") in valid_tasks
        assert is_valid_task, \
            f"'{model_nickname}' requires an image segmentation task."
        return SwinUNETRModel(cfg=cfg)

    else:
        raise ValueError(f"Unknown model type '{model_nickname}'. Valid values are {[t.value for t in ModelType]}.")
