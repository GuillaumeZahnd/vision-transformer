import torch
import torch.nn as nn

from patch_embeddings import PatchEmbedding
from vision_transformer_block import VisionTransformerBlock


class VisionTransformerModel(nn.Module):
    def __init__(
        self,
        nb_channels: int,
        image_height: int,
        image_width: int,
        patch_side_length: int,
        stride: int,
        nb_layers: int,
        embedding_dim: int,
        nb_heads: int,
        mlp_expansion: int,
        nb_classes: int) -> None:

        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, embedding_dim))

        self.embed_patches = PatchEmbedding(
            nb_channels=nb_channels,
            image_height=image_height,
            image_width=image_width,
            patch_side_length=patch_side_length,
            embedding_dim=embedding_dim,
            stride=stride)

        self.transformer_layers = \
            nn.ModuleList([
                VisionTransformerBlock(
                    embedding_dim=embedding_dim,
                    nb_heads=nb_heads,
                    nb_patches_height=self.embed_patches.get_nb_patches_height(),
                    nb_patches_width=self.embed_patches.get_nb_patches_width(),
                    mlp_expansion=mlp_expansion)
                for _ in range(nb_layers)])

        self.multilayer_perceptron_classification_head = nn.Sequential(
            nn.Linear(embedding_dim, nb_classes),
            nn.Softmax(dim=-1))


    def forward(self, input_images):
        """
        Apply the vision transformer encoder for image classification.

        Args:
            Batch of images, in the shape (batch_size, nb_channels, image_height, image_width).

        Returns:
            Class probabilities, in the shape (batch_size, nb_classes).
        """

        batch_size = input_images.shape[0]
        patch_embeddings = self.embed_patches(images=input_images)
        cls_embeddings = self.cls_token.expand(batch_size, -1, -1)
        token_embeddings = torch.cat((cls_embeddings, patch_embeddings), dim=1)

        for layer in self.transformer_layers:
            token_embeddings = layer(token_embeddings)

        # Utilize the cls slice (at index zero) to represent the token considered for the classification task
        class_token_output = token_embeddings[:, 0, :]

        class_probabilities = self.multilayer_perceptron_classification_head(class_token_output)

        return class_probabilities
