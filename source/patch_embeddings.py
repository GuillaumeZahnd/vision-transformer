import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        nb_channels: int,
        image_height: int,
        image_width: int,
        patch_side_length: int,
        embedding_dim: int,
        stride: int) -> None:

        super().__init__()

        assert stride >= 1 and stride <= patch_side_length, "The stride must be comprised in [1, patch_side_length]."

        self._nb_patches_height = 1 + (image_height - patch_side_length) // stride
        self._nb_patches_width = 1 + (image_width - patch_side_length) // stride

        self.conv2d = nn.Conv2d(
            in_channels=nb_channels,
            out_channels=embedding_dim,
            kernel_size=patch_side_length,
            stride=stride)


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert an image into a patch embeddings using a two-dimensional convolutional layer.

        Args:
            Batch of images, in the shape (batch_size, nb_channels, image_height, image_width).

        Returns:
            Batch of patch embeddings, in the shape (batch_size, nb_patches, embedding_dim).
        """
        return self.conv2d(images).flatten(2).transpose(1, 2)


    def get_nb_patches_height(self) -> int:
        """
        Number of patches along the height of the image.
        """
        return self._nb_patches_height


    def get_nb_patches_width(self) -> int:
        """
        Number of patches along the width of the image.
        """
        return self._nb_patches_width


    def get_nb_patches_total(self) -> int:
        """
        Number of patches in the image.
        """
        return self._nb_patches_height * self._nb_patches_width
