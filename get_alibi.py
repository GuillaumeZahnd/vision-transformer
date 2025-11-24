import torch
import math
import itertools
from typing import List


def get_alibi(number_of_heads: int, number_of_patches_per_side: int) -> torch.Tensor:
    """
    Compute the linear biases required for ALiBi.
    See: Press et al., Train Short, Test Long, ICLR 2022 (https://arxiv.org/pdf/2108.12409).

    Args:
        number_of_heads: Number of attention heads.
        number_of_patches_per_side: Number of patches along each side of the image.

    Returns:
        Linear biases, in the shape (1, number_of_heads, number_of_patches_per_side +1, number_of_patches_per_side +1).
    """

    def get_slope(number_of_heads: int) -> torch.Tensor:
        """
            Compute a slope, which represents decreasing head-specific coefficients.
            Inspired from https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742

            Args:
                number_of_heads: Number of attention heads.

            Returns:
                Slope, in the shape (number_of_heads).
        """

        def get_slope_power_of_2(number_of_heads: int) -> List[int]:
            """
            Compute the slope coefficients, as a geometric series starting from O.5^8.

            Args:
                number_of_heads: Number of attention heads.

            Returns:
                List of slope coefficients, of length (number_of_heads).

            """

            # The starting value is effectively "2^-(8/number_of_heads)", up to numerical precision.
            start = (2 ** (-2 ** -(math.log2(number_of_heads) - 3)))

            ratio = start
            return [start * ratio ** i for i in range(number_of_heads)]

        if math.log2(number_of_heads).is_integer():
            slope = get_slope_power_of_2(number_of_heads)

        else:
            closest_smaller_power_of_2 = 2 ** math.floor(math.log2(number_of_heads))
            slope = \
                get_slope_power_of_2(closest_smaller_power_of_2) + \
                get_slope(2 * closest_smaller_power_of_2)[0::2][:number_of_heads - closest_smaller_power_of_2]

        slope = torch.tensor(slope, dtype=torch.float32)
        slope, _ = torch.sort(slope, dim=0, descending=True)
        return slope

    slope = get_slope(number_of_heads=number_of_heads)


    def get_distances(number_of_patches_per_side: int) -> torch.Tensor:
        """
        Calculate the Euclidean distance between pairs of patches for all possible combinations.

        Args:
            number_of_patches_per_side: Number of patches per side.

        Returns:
            Distances, in the shape (number_of_patches_per_side^2, number_of_patches_per_side^2).
        """

        points = list(itertools.product(range(number_of_patches_per_side), range(number_of_patches_per_side)))
        points = torch.tensor(points, dtype=torch.float32)
        points_vertical = points.unsqueeze(0)
        points_horizontal = points.unsqueeze(1)
        distances = torch.squeeze(torch.cdist(points_vertical, points_horizontal))
        return distances

    distances = get_distances(number_of_patches_per_side=number_of_patches_per_side)


    def pad_left_and_top_for_cls_token(x: torch.Tensor) -> torch.Tensor:
        """
            Pad the "leftmost column" and "topmost row" of the distance tensor to account for the <cls> token.
            The padded value is zero, because <cls> can attend to all patches with a distance of zero.

            Args:
                Distance tensor, in the shape (number_of_patches_per_side, number_of_patches_per_side).

            Returns:
                Padded distance tensor, in the shape (number_of_patches_per_side +1, number_of_patches_per_side +1).
        """
        return torch.nn.functional.pad(input=x, pad=(1, 0, 1, 0), value=0.0)

    distances = pad_left_and_top_for_cls_token(x=distances)


    alibi = -1 * distances.unsqueeze(0).unsqueeze(1) * slope.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return alibi
