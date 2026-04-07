import torch
import torch.nn as nn


class VisionTransformerSegmentationBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        nb_classes: int,
        nb_feature_maps: int,
        image_height: int,
        image_width: int
    ) -> None:
        """
        Transformer segmentation block.

        Args:
            embedding_dim: Dimensionality of the encoder's output features.
            nb_classes: Number of pixel-level segmentation classes, including the background.
            nb_feature_maps: Number of intermediate encoder layers extracted for fusion.
            image_height: Size of the original image along the vertical dimension.
            image_width: Size of the original image along the horizontal dimension.
        """
        super().__init__()

        self.image_height = image_height
        self.image_width = image_width

        # Perform learned weighted average by reducing from (B, D*nb_intermediate_layers, H, W) to (B, D, H, W)
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(embedding_dim * nb_feature_maps, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        # Maps embeddings to semantic classes
        self.classifier = nn.Conv2d(embedding_dim, nb_classes, kernel_size=1)


    def forward(self, multi_scale_feature_maps: list[torch.Tensor]) -> torch.Tensor:
        """
        Transformer segmentation pathway, mapping a list of latent features to pixel-level semantic labels.

        Args:
            multi_scale_feature_maps: List of latent features, of shape (B, D, H_patch, W_patch) and of length nb_feature_maps.

        Returns:
            Segmentation logits, of shape (batch, nb_classes, image_height, image_width).
        """
        # Concatenate along the channel dimension (dim=1)
        # [(B, D, H_patch, W_patch), (B, D, H_patch, W_patch), ...] -> (B, D * nb_feature_maps, H_patch, W_patch)
        x = torch.cat(multi_scale_feature_maps, dim=1)

        # Reduce dimensionality
        # (B, D * nb_feature_maps, H_patch, W_patch) -> (B, D, H_patch, W_patch)
        x = self.feature_fusion(x)

        # Predict class logits
        # (B, D, H_patch, W_patch) -> (B, nb_classes, H_patch, W_patch)
        logits = self.classifier(x)

        # Upsample to original image resolution
        # (B, nb_classes, H_patch, W_patch) -> (B, nb_classes, image_height, image_width)
        out = nn.functional.interpolate(
            logits,
            size=(self.image_height, self.image_width),
            mode="bilinear",
            align_corners=False
        )

        return out
