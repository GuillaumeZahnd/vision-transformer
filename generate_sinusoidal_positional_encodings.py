import torch


def generate_sinusoidal_positional_encodings(
    sequence_length: int, embedding_dimension: int, base_wavelength: int=1e5) -> torch.Tensor:

    positional_encodings = torch.zeros(sequence_length, embedding_dimension)
    denominator = base_wavelength ** (torch.arange(0, embedding_dimension, 2).float() / embedding_dimension)
    positions = torch.arange(sequence_length).unsqueeze(1).float()
    positional_encodings[:, 0::2] = torch.sin(positions / denominator)
    positional_encodings[:, 1::2] = torch.cos(positions / denominator)

    return positional_encodings
