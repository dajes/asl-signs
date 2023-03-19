import torch


def preprocess(features: torch.Tensor, max_len: int):
    """

    Args:
        features: Tensor of shape (seq_len, n_points, 3)
        max_len: Maximum sequence length

    Returns:
        features: Tensor of shape (seq_len, n_relevant_points, 3)

    """
    features[torch.isnan(features)] = 0
    return features
