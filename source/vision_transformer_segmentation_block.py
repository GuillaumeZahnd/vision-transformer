import torch
import torch.nn as nn


class VisionTransformerSegmentationBlock(nn.Module):
    def __init__(self, embedding_dim: int, nb_semantic_classes: int, nb_feature_maps: int, patch_size: int) -> None:
        """
        Transformer segmentation block.

        Args:
            embedding_dim: Dimensionality of the encoder's output features.
            nb_semantic_classes: Number of pixel-level segmentation classes, including the background.
            nb_feature_maps: Number of intermediate encoder layers extracted for fusion.
            patch_size: Side length of each patch, in pixels. Patches are square.
        """
        super().__init__()

        self.patch_size = patch_size

        # Perform learned weighted average by reducing from (B, D*nb_intermediate_layers, H, W) to (B, D, H, W)
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(embedding_dim * nb_feature_maps, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        # Maps embeddings to semantic classes
        self.classifier = nn.Conv2d(embedding_dim, nb_semantic_classes, kernel_size=1)


    def forward(self, multi_scale_feature_maps: list[torch.Tensor]) -> torch.Tensor:
        """
        Transformer segmentation pathway, mapping a list of latent features to pixel-level semantic labels.

        Args:
            multi_scale_feature_maps: List of latent features [(B,D,H,W), (B,D,H,W), ...], of length nb_feature_maps.
        """
        # Concatenate along the channel dimension (dim=1)
        x = torch.cat(multi_scale_feature_maps, dim=1)

        # Reduce dimensionality
        x = self.feature_fusion(x)

        # Predict class logits
        logits = self.classifier(x)

        # Upsample to original image resolution
        out = nn.functional.interpolate(
            logits,
            scale_factor=self.patch_size,
            mode="bilinear",
            align_corners=False
        )

        return out
