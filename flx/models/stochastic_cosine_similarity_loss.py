import torch


class StochasticCosineSimilarityLoss(torch.nn.Module):
    """
    Stochastic cosine similarity loss.
    Inspired by ArcFace loss

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        alpha: "learning rate" of the centers. For each datapoint centers are updated by center + alpha * (datapoint - center)
    """

    def __init__(self, num_classes: int, feat_dim: int, alpha: float = 0.05):
        super(StochasticCosineSimilarityLoss, self).__init__()
        self.alpha = alpha
        self.register_buffer(
            "centers",
            torch.nn.functional.normalize(torch.randn(num_classes, feat_dim), dim=1),
            persistent=True,
        )
        self.register_buffer(
            "update",
            torch.clone(self.centers),
            persistent=True,
        )
        self.counter: int = 1

    def forward(self, x: torch.Tensor, labels: torch.LongTensor):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # Copy the centers into temporary matrix
        with torch.no_grad():
            batch_centers = torch.index_select(self.centers, 0, labels)
            label_mat = labels.reshape((1, -1)).repeat(labels.shape[0], 1)
            target_similarities = (label_mat == label_mat.transpose(0, 1)).float()
        # Similarity to own class center should be one, similarity to other class centers should be zero
        similarity = torch.nn.functional.relu(
            torch.einsum("pd,td->pt", x, batch_centers)
        )
        diff = similarity - target_similarities
        # Update current centers
        with torch.no_grad():
            self.update.index_add_(0, labels, x)
            if self.counter % 32 == 0:
                self.centers = self.centers * (self.alpha * self.update)
                self.centers = torch.nn.functional.normalize(self.centers, dim=1)
                self.update = torch.clone(self.centers)
                self.counter = 0
            self.counter += 1
        # Return SSD
        return torch.sum(diff**2)
