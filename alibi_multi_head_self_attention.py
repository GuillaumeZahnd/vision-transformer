from get_alibi import get_alibi

import torch
import torch.nn as nn
import math


class AlibiMultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim:int, nb_heads:int, nb_patches_height: int, nb_patches_width: int) -> None:
        super().__init__()

        assert embedding_dim % nb_heads == 0, "The embedding dimension must be a multiple of the number of heads."

        self.nb_heads = nb_heads
        self.head_dim = embedding_dim // nb_heads

        self.q_weights = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_weights = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_weights = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.linear = nn.Linear(embedding_dim, embedding_dim)

        # ALiBi is used for positional encoding
        alibi, _, _ = get_alibi(nb_heads=self.nb_heads, nb_patches_height=nb_patches_height, nb_patches_width=nb_patches_width)
        self.register_buffer("alibi", alibi, persistent=False)


    def forward(self, sequence_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute the multi-head self attention, with the ALiBi positional encoding.
        See: Press et al., Train Short, Test Long, ICLR 2022 (https://arxiv.org/pdf/2108.12409).

        Args:
            sequence_embedding: Input sequence embedding, in the shape (batch_size, nb_patches_height*nb_patches_width+1, embedding_dim).

        Returns:
            Weighted sum of values, in the shape (batch_size, nb_patches_height*nb_patches_width+1, embedding_dim).
            Attention weights, in the shape (batch_size, nb_heads, nb_patches_height*nb_patches_width+1, nb_patches_height*nb_patches_width+1).
        """

        batch_size, sequence_length, embedding_dim = sequence_embedding.shape

        q = self.q_weights(sequence_embedding).view(batch_size, sequence_length, self.nb_heads, self.head_dim).transpose(1, 2)
        k = self.k_weights(sequence_embedding).view(batch_size, sequence_length, self.nb_heads, self.head_dim).transpose(1, 2)
        v = self.v_weights(sequence_embedding).view(batch_size, sequence_length, self.nb_heads, self.head_dim).transpose(1, 2)

        scaled_attention_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        # Crucially, the positional encoding is applied here
        attention_weights = torch.softmax(scaled_attention_logits + self.alibi, dim=-1)

        weighted_values = torch.matmul(attention_weights, v)
        weighted_values = weighted_values.transpose(1, 2).contiguous().view((batch_size, sequence_length, embedding_dim))
        weighted_values = self.linear(weighted_values)

        return weighted_values, attention_weights
