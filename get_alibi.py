import torch
import math
import itertools


def get_alibi(nb_heads: int, nb_patches_per_side: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the linear biases required for ALiBi.
    See: Press et al., Train Short, Test Long, ICLR 2022 (https://arxiv.org/pdf/2108.12409).

    Args:
        nb_heads: Number of attention heads.
        nb_patches_per_side: Number of patches along each side of the image.

    Returns:
        Linear biases, in the shape (1, nb_heads, nb_patches_per_side +1, nb_patches_per_side +1).
        Distances between patches, in the shape (nb_patches_per_side^2 +1, nb_patches_per_side^2 +1)
        Head-specific slope, in the shape (nb_heads)
    """

    def get_slope(nb_heads: int) -> torch.Tensor:
        """
            Compute a slope, which represents decreasing head-specific coefficients.
            Inspired from https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742

            Args:
                nb_heads: Number of attention heads.

            Returns:
                Slope, in the shape (nb_heads).
        """

        def get_slope_power_of_2(nb_heads: int) -> torch.Tensor:
            """
            Compute the slope coefficients, as a geometric series starting from O.5^8.

            Args:
                nb_heads: Number of attention heads.

            Returns:
                List of slope coefficients, of length (nb_heads).

            """

            # The starting value is effectively "2^-(8/nb_heads)", up to numerical precision.
            start = (2 ** (-2 ** -(math.log2(nb_heads) - 3)))

            ratio = start
            slope = [start * ratio ** i for i in range(nb_heads)]
            slope = torch.tensor(slope, dtype=torch.float32)
            return slope

        if math.log2(nb_heads).is_integer():
            slope = get_slope_power_of_2(nb_heads)

        else:
            closest_smaller_power_of_2 = 2 ** math.floor(math.log2(nb_heads))
            slope_lower = get_slope_power_of_2(closest_smaller_power_of_2)
            slope_higher = get_slope(2 * closest_smaller_power_of_2)[0::2][:nb_heads - closest_smaller_power_of_2]
            slope = torch.cat((slope_lower, slope_higher))
            slope, _ = torch.sort(slope, dim=0, descending=True)

        return slope

    slope = get_slope(nb_heads=nb_heads)


    def get_distances(nb_patches_per_side: int) -> torch.Tensor:
        """
        Calculate the Euclidean distance between pairs of patches for all possible combinations.

        Args:
            nb_patches_per_side: Number of patches per side.

        Returns:
            Distances, in the shape (nb_patches_per_side^2, nb_patches_per_side^2).
        """

        points = list(itertools.product(range(nb_patches_per_side), range(nb_patches_per_side)))
        points = torch.tensor(points, dtype=torch.float32)
        points_vertical = points.unsqueeze(0)
        points_horizontal = points.unsqueeze(1)
        distances = torch.squeeze(torch.cdist(points_vertical, points_horizontal))
        return distances

    distances = get_distances(nb_patches_per_side=nb_patches_per_side)


    def pad_left_and_top_for_cls_token(x: torch.Tensor) -> torch.Tensor:
        """
            Pad the "leftmost column" and "topmost row" of the distance tensor to account for the <cls> token.
            The padded value is zero, because <cls> can attend to all patches with a distance of zero.

            Args:
                Distance tensor, in the shape (nb_patches_per_side, nb_patches_per_side).

            Returns:
                Padded distance tensor, in the shape (nb_patches_per_side +1, nb_patches_per_side +1).
        """
        return torch.nn.functional.pad(input=x, pad=(1, 0, 1, 0), value=0.0)

    distances = pad_left_and_top_for_cls_token(x=distances)


    alibi = -1 * distances.unsqueeze(0).unsqueeze(1) * slope.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return alibi, distances, slope
