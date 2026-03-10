import torch
import torch.nn as nn
from enum import Enum
from omegaconf import DictConfig

from source.patch_embeddings import PatchEmbedding
from source.vision_transformer_encoder_block import VisionTransformerEncoderBlock
from source.vision_transformer_segmentation_block import VisionTransformerSegmentationBlock


class ModelMode(Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"


def _init_weights_kaiming_he(m: nn.Module) -> None:
    """
    Kaiming He initialization for linear layers.
    Ideal for models using ReLU or GELU activations.
    Keep activation variance constant (~1.0) across layers; this is more critical than the actual values themselves.
    Gradient survival: Prevent degeneration during the first epoch (neither vanishing, nor exploding).
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class VisionTransformerModel(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:

        super().__init__()

        self.model_type = cfg.model.model_type
        self.embedding_dim = cfg.model.embedding_dim

        self.cls_token = nn.Parameter(torch.randn(1, cfg.model.embedding_dim))

        self.embed_patches = PatchEmbedding(
            nb_channels=cfg.dataset.nb_channels,
            image_height=cfg.dataset.image_height,
            image_width=cfg.dataset.image_width,
            patch_side_length=cfg.model.patch_side_length,
            embedding_dim=cfg.model.embedding_dim,
            stride=cfg.model.stride
        )

        self.transformer_layers = \
            nn.ModuleList([
                VisionTransformerEncoderBlock(
                    embedding_dim=cfg.model.embedding_dim,
                    nb_heads=cfg.model.nb_heads,
                    nb_patches_height=self.embed_patches.nb_patches_height,
                    nb_patches_width=self.embed_patches.nb_patches_width,
                    mlp_expansion=cfg.model.mlp_expansion)
                for _ in range(cfg.model.nb_layers)]
        )

        # This is the step before the cross-entropy loss, which applies a softmax
        self.multilayer_perceptron_classification_head = nn.Linear(cfg.model.embedding_dim, cfg.dataset.nb_classes)

        if self.model_type == ModelMode.SEGMENTATION.value:
            self.decoder = VisionTransformerSegmentationBlock(
                embedding_dim=cfg.model.embedding_dim,
                nb_classes=cfg.dataset.nb_semantic_labels,
                nb_feature_maps=len(cfg.model.layers_for_segmentation),
                patch_size=cfg.model.patch_side_length
            )

        self.apply(_init_weights_kaiming_he)


    def forward(self, input_images: torch.Tensor):
        """
        Process images through the Vision Transformer for either classification or segmentation.

        Args:
            Batch of images, in the shape (batch_size, nb_channels, image_height, image_width).

        Returns:
            - If classification: Class logits, of shape (batch_size, nb_classes).
            - If segmentation: Semantic segmentation mask, of shape (batch_size, nb_semantic_classes, image_height, image_width).
        """

        batch_size = input_images.shape[0]
        patch_embeddings = self.embed_patches(images=input_images)
        cls_embeddings = self.cls_token.expand(batch_size, -1, -1)
        token_embeddings = torch.cat((cls_embeddings, patch_embeddings), dim=1)

        # For segmentation tasks only
        multi_scale_feature_maps = []

        for layer_id, layer in enumerate(self.transformer_layers):
            token_embeddings = layer(token_embeddings)

            if self.model_type == "segmentation" and layer_id in cfg.model.layers_for_segmentation:
                latent = token_embeddings[:, 1:, :]  # Exclude the CLS token
                latent = latent.transpose(1, 2)  # Transpose (B, N, D) -> (B, D, N)
                latent = latent.reshape(  # Reshape (B, D, N) -> (B, D, H, W)
                    batch_size,
                    self.embedding_dim,
                    self.embed_patches.nb_patches_height,
                    self.embed_patches.nb_patches_width
                )
                multi_scale_feature_maps.append(latent)

        if self.model_type == ModelMode.CLASSIFICATION.value:
            # Utilize the cls slice (at index zero) to represent the token considered for the classification task
            image_class_token_output = token_embeddings[:, 0, :]
            logits = self.multilayer_perceptron_classification_head(image_class_token_output)
            return logits

        elif self.model_type == ModelMode.SEGMENTATION.value:
            segmented_image = self.decoder(multi_scale_feature_maps)
            return segmented_image

        else:
            raise ValueError("Unknown model type '{}'. Valid values are {}.".format(self.model_type, [e.value for e in ModelMode]))
