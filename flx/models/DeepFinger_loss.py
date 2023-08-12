import torch
from torch import nn

from flx.models.DeepFinger import DeepFingerTrainingOutput
from flx.models.center_loss import CenterLoss

W_CROSS_ENTROPY = 1.0
W_CENTER_LOSS = 0.125
W_MINUTIA_MAP_LOSS = 0.3


def _compute_minutia_map_loss(
    x: torch.Tensor,
    minutia_maps: torch.Tensor,
    minutia_map_weights: torch.Tensor,
) -> torch.Tensor:
    mm_squared_diff = (x - minutia_maps) ** 2
    mm_mse = mm_squared_diff.reshape(minutia_map_weights.shape[0], -1).mean(dim=1)
    return (mm_mse * minutia_map_weights).mean()


class _DeepFinger_Embedding_Loss(nn.Module):
    def __init__(self, num_classes: int, num_embedding_dims: int):
        """
        @param center_loss_weight : Must be greater than zero. The center loss is multiplied by this term.
        """
        super().__init__()
        self.crossent_loss_fun = nn.CrossEntropyLoss()
        self.center_loss_fun = CenterLoss(num_classes, num_embedding_dims)
        self.crossent_loss_sum: float = 0
        self.center_loss_sum: float = 0
        self.n_datapoints: int = 0

    def reset_recorded_loss(self) -> None:
        self.crossent_loss_sum = 0
        self.center_loss_sum = 0

    def get_recorded_loss(self) -> dict:
        if self.n_datapoints == 0:
            return {"crossent_loss_sum": 0, "center_loss_sum": 0}
        return {
            "crossent_loss_sum": self.crossent_loss_sum / self.n_datapoints,
            "center_loss_sum": self.center_loss_sum / self.n_datapoints,
        }

    def forward(
        self, embeddings: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        crossent_loss = self.crossent_loss_fun(logits, labels)
        center_loss = self.center_loss_fun(embeddings, labels)
        self.crossent_loss_sum += float(crossent_loss) * W_CROSS_ENTROPY
        self.center_loss_sum += float(center_loss) * W_CENTER_LOSS
        self.n_datapoints += labels.shape[0]
        return W_CROSS_ENTROPY * crossent_loss + W_CENTER_LOSS * center_loss


class DeepFingerLoss_Tex(nn.Module):
    """
    The weights that are use for the

    """

    def __init__(self, num_classes: int, texture_embedding_dims: int):
        """
        @param center_loss_weight : Must be greater than zero. The center loss is multiplied by this term.
        """
        super().__init__()
        self.texture_loss_fun: _DeepFinger_Embedding_Loss = _DeepFinger_Embedding_Loss(
            num_classes=num_classes, num_embedding_dims=texture_embedding_dims
        )

    def forward(
        self,
        output: DeepFingerTrainingOutput,
        labels: torch.Tensor,
        minutia_maps: torch.Tensor,
        minutia_map_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        @param output : Output of the model for the given fingerprint images; Object of type DeepFingerTrainingOutput
        @param labels : Ground truth subject ids for fingerprints images
        @param minutia_maps : Ground truth minutia maps for the fingerprint images
        """
        if minutia_maps.shape[1] != 0:
            raise RuntimeWarning(
                "DeepFinger_TextureLoss received non-empty minutia_maps!"
            )
        texture_loss = self.texture_loss_fun(
            output.texture_embeddings, output.texture_logits, labels
        )
        return texture_loss

    def get_recorded_loss(self) -> dict:
        return self.texture_loss_fun.get_recorded_loss()

    def reset_recorded_loss(self) -> None:
        self.texture_loss_fun.reset_recorded_loss()


class DeepFingerLoss_Minu(nn.Module):

    def __init__(
        self,
        num_classes: int,
        minutia_embedding_dims: int,
    ):
        """
        @param center_loss_weight : Must be greater than zero. The center loss is multiplied by this term.
        """
        super().__init__()
        self.minu_loss_fun: _DeepFinger_Embedding_Loss = _DeepFinger_Embedding_Loss(
            num_classes=num_classes,
            num_embedding_dims=minutia_embedding_dims,
        )
        self.minu_map_loss_sum: float = 0

    def forward(
        self,
        output: DeepFingerTrainingOutput,
        labels: torch.Tensor,
        minutia_maps: torch.Tensor,
        minutia_map_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        @param output : Output of the model for the given fingerprint images; Object of type DeepFingerTrainingOutput
        @param labels : Ground truth subject ids for fingerprints images
        @param minutia_maps : Ground truth minutia maps for the fingerprint images
        """
        minutia_loss = self.minu_loss_fun(
            output.minutia_embeddings, output.minutia_logits, labels
        )
        mm_loss = _compute_minutia_map_loss(
            output.minutia_maps, minutia_maps, minutia_map_weights
        )
        self.minu_map_loss_sum += W_MINUTIA_MAP_LOSS * float(mm_loss)
        return minutia_loss + W_MINUTIA_MAP_LOSS * mm_loss

    def get_recorded_loss(self) -> dict:
        return {
            "minutia_loss": self.minu_loss_fun.get_recorded_loss(),
            "minutia_map_loss": self.minu_map_loss_sum
            / self.minu_loss_fun.n_datapoints
            if self.minu_loss_fun.n_datapoints > 0
            else 0,
        }

    def reset_recorded_loss(self) -> None:
        self.minu_loss_fun.reset_recorded_loss()
        self.minu_map_loss_sum = 0


class DeepFingerLoss_TexMinu(nn.Module):
    """
    The weights that are use for the

    """

    def __init__(
        self,
        num_classes: int,
        texture_embedding_dims: int,
        minutia_embedding_dims: int,
    ):
        """
        @param center_loss_weight : Must be greater than zero. The center loss is multiplied by this term.
        """
        super().__init__()
        self.minu_loss_fun: _DeepFinger_Embedding_Loss = _DeepFinger_Embedding_Loss(
            num_classes=num_classes,
            num_embedding_dims=texture_embedding_dims,
        )
        self.texture_loss_fun: _DeepFinger_Embedding_Loss = _DeepFinger_Embedding_Loss(
            num_classes=num_classes,
            num_embedding_dims=minutia_embedding_dims,
        )
        self.minu_map_loss_sum: float = 0

    def forward(
        self,
        output: DeepFingerTrainingOutput,
        labels: torch.Tensor,
        minutia_maps: torch.Tensor,
        minutia_map_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        @param output : Output of the model for the given fingerprint images; Object of type DeepFingerTrainingOutput
        @param labels : Ground truth subject ids for fingerprints images
        @param minutia_maps : Ground truth minutia maps for the fingerprint images
        """
        minutia_loss = self.minu_loss_fun(
            output.minutia_embeddings, output.minutia_logits, labels
        )
        texture_loss = self.texture_loss_fun(
            output.texture_embeddings, output.texture_logits, labels
        )
        mm_loss = _compute_minutia_map_loss(
            output.minutia_maps, minutia_maps, minutia_map_weights
        )
        self.minu_map_loss_sum += W_MINUTIA_MAP_LOSS * float(mm_loss)
        return texture_loss + minutia_loss + W_MINUTIA_MAP_LOSS * mm_loss

    def get_recorded_loss(self) -> dict:
        return {
            "texture_loss": self.texture_loss_fun.get_recorded_loss(),
            "minutia_loss": self.minu_loss_fun.get_recorded_loss(),
            "minutia_map_loss": self.minu_map_loss_sum
            / self.texture_loss_fun.n_datapoints
            if self.texture_loss_fun.n_datapoints > 0
            else 0,
        }

    def reset_recorded_loss(self) -> None:
        self.texture_loss_fun.reset_recorded_loss()
        self.minu_loss_fun.reset_recorded_loss()
        self.minu_map_loss_sum = 0


class DeepFingerLoss_TexMinuCombi(nn.Module):
    """
    The weights that are use for the

    """

    def __init__(
        self,
        num_classes: int,
        texture_embedding_dims: int,
        minutia_embedding_dims: int,
    ):
        """
        @param center_loss_weight : Must be greater than zero. The center loss is multiplied by this term.
        """
        super().__init__()
        self.combi_loss_fun: _DeepFinger_Embedding_Loss = _DeepFinger_Embedding_Loss(
            num_classes=num_classes,
            num_embedding_dims=texture_embedding_dims + minutia_embedding_dims,
        )
        self.minu_map_loss_sum: float = 0

    def forward(
        self,
        output: DeepFingerTrainingOutput,
        labels: torch.Tensor,
        minutia_maps: torch.Tensor,
        minutia_map_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        @param output : Output of the model for the given fingerprint images; Object of type DeepFingerTrainingOutput
        @param labels : Ground truth subject ids for fingerprints images
        @param minutia_maps : Ground truth minutia maps for the fingerprint images
        """
        combi_loss = self.combi_loss_fun(
            torch.concatenate((output.texture_embeddings, output.minutia_embeddings), dim=1), output.combined_logits, labels
        )
        mm_loss = _compute_minutia_map_loss(
            output.minutia_maps, minutia_maps, minutia_map_weights
        )
        self.minu_map_loss_sum += W_MINUTIA_MAP_LOSS * float(mm_loss)
        return 2 * combi_loss + W_MINUTIA_MAP_LOSS * mm_loss

    def get_recorded_loss(self) -> dict:
        return {
            "combi_loss": self.combi_loss_fun.get_recorded_loss(),
            "minutia_map_loss": self.minu_map_loss_sum
            / self.combi_loss_fun.n_datapoints
            if self.combi_loss_fun.n_datapoints > 0
            else 0,
        }

    def reset_recorded_loss(self) -> None:
        self.combi_loss_fun.reset_recorded_loss()
        self.minu_map_loss_sum = 0


def main():
    loss = DeepFingerLoss_Tex(100)
    loss = DeepFingerLoss_TexMinu(100)
    print("No syntax errors")


if __name__ == "__main__":
    main()
