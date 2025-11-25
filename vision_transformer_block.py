import torch
import torch.nn as nn

from alibi_multi_head_self_attention import AlibiMultiHeadSelfAttention


class VisionTransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, nb_heads: int, nb_patches_height: int, nb_patches_width: int, mlp_expansion: int=4) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = nb_heads

        self.normalization_layer_1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.normalization_layer_2 = nn.LayerNorm(normalized_shape=embedding_dim)

        self.attention_layer = AlibiMultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            nb_heads=nb_heads,
            nb_patches_height=nb_patches_height,
            nb_patches_width=nb_patches_width)

        self.feedforward_block = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_expansion * embedding_dim),
            nn.GELU(),
            nn.Linear(in_features=mlp_expansion * embedding_dim, out_features=embedding_dim))


    def _residual_block_1(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Residual block, combining a pre-layer normalization followed by an attention mechanism.

        Args:
            input_embeddings: Input embeddings, in the shape (batch_size, nb_patches_height*nb_patches_width+1, embedding_dim).

        Returns:
            Residual embeddings, in the shape (batch_size, nb_patches_height*nb_patches_width+1, embedding_dim).
        """

        normalized_embeddings = self.normalization_layer_1(input_embeddings)
        attention_embeddings, _ = self.attention_layer(sequence_embedding=normalized_embeddings)
        residual_embeddings = input_embeddings + attention_embeddings
        return residual_embeddings


    def _residual_block_2(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Residual block, combining a pre-layer normalization followed by a multilayer perceptron.

        Args:
            input_embeddings: Input embeddings, in the shape (batch_size, nb_patches_height*nb_patches_width+1, embedding_dim).

        Returns:
            Residual embeddings, in the shape (batch_size, nb_patches_height*nb_patches_width+1, embedding_dim).
        """

        normalized_embeddings = self.normalization_layer_2(input_embeddings)
        feedforward_embeddings = self.feedforward_block(normalized_embeddings)
        residual_embeddings = input_embeddings + feedforward_embeddings
        return residual_embeddings


    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Transformer block, combining two residual blocks: an attention block followed by a multilayer perceptron block.

        Args:
            input_embeddings: Input embeddings, in the shape (batch_size, nb_patches_height*nb_patches_width+1, embedding_dim).

        Returns:
            Output embeddings, in the shape (batch_size, nb_patches_height*nb_patches_width+1, embedding_dim).
        """

        output_embeddings = self._residual_block_1(input_embeddings=input_embeddings)
        output_embeddings = self._residual_block_2(input_embeddings=output_embeddings)
        return output_embeddings
