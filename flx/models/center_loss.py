from os.path import join

import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        alpha: "learning rate" of the centers. For each datapoint centers are updated by center + alpha * (datapoint - center)
    """

    def __init__(self, num_classes: int, feat_dim: int, alpha: float = 0.01):
        super(CenterLoss, self).__init__()
        self.alpha = alpha
        self.register_buffer(
            "centers",
            torch.nn.functional.normalize(torch.randn(num_classes, feat_dim), dim=1),
            persistent=True,
        )
        self.counter = 0
        self.nupdate = 0

    def forward(self, x: torch.Tensor, labels: torch.LongTensor):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # Copy the centers into temporary matrix
        with torch.no_grad():
            batch_centers = torch.index_select(self.centers, 0, labels)
        # Get difference of x from batch centers
        diff = x - batch_centers
        # Update current centers
        with torch.no_grad():
            self.centers.index_add_(0, labels, diff, alpha=self.alpha)
            # self._disperse_centers()
        # Return mean loss
        return torch.sum(diff**2)
