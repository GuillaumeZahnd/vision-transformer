from omegaconf import DictConfig
from enum import Enum
import torch


from source.models.vision_transformer_model import VisionTransformerModel
from source.models.efficientnet_model import EfficientNetModel


class ModelType(Enum):
    CUSTOM_VIT = "custom_vit"
    EFFICIENTNET = "efficientnet"
    
    
def select_model(cfg: DictConfig) -> torch.nn:
    model_nickname = cfg.model.model_type

    if model_nickname == ModelType.CUSTOM_VIT.value:
        return VisionTransformerModel(cfg=cfg)

    elif model_nickname == ModelType.EFFICIENTNET.value:
        return EfficientNetModel(cfg=cfg)

    else:
        raise ValueError(f"Unknown model type '{model_nickname}'. Valid values are {[t.value for t in ModelType]}.")
