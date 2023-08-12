from flx.models import InceptionV4
import torch
from torch import nn

from flx.models.localization_network import LocalizationNetwork

"""
Terminology:

Each subject in out biometric dataset (i.e. each fingerprint)
corresponds to one class in the logits that are calculated from the embedding.

We employ a loss function called the "center loss" (see the corresponding class)
that encourages the model to generate similar embeddings for samples from the
same subject.

"""

DEEPPRINT_INPUT_SIZE = 299

"""
  _ __ ___   ___   __| | ___| |   ___ ___  _ __ ___  _ __   ___  _ __   ___ _ __ | |_ ___ 
 | '_ ` _ \ / _ \ / _` |/ _ \ |  / __/ _ \| '_ ` _ \| '_ \ / _ \| '_ \ / _ \ '_ \| __/ __|
 | | | | | | (_) | (_| |  __/ | | (_| (_) | | | | | | |_) | (_) | | | |  __/ | | | |_\__ \ 
 |_| |_| |_|\___/ \__,_|\___|_|  \___\___/|_| |_| |_| .__/ \___/|_| |_|\___|_| |_|\__|___/
                                                    |_|                                   
"""


class _InceptionV4_Stem(nn.Module):
    def __init__(self):
        super(_InceptionV4_Stem, self).__init__()
        # Modules
        self.features = nn.Sequential(
            InceptionV4.BasicConv2d(1, 32, kernel_size=3, stride=2),
            InceptionV4.BasicConv2d(32, 32, kernel_size=3, stride=1),
            InceptionV4.BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            InceptionV4.Mixed_3a(),
            InceptionV4.Mixed_4a(),
            InceptionV4.Mixed_5a(),
        )

    def forward(self, input):
        assert input.shape[-1] == DEEPPRINT_INPUT_SIZE
        assert input.shape[-2] == DEEPPRINT_INPUT_SIZE
        x = self.features(input)
        return x


class _Branch_TextureEmbedding(nn.Module):
    def __init__(self, texture_embedding_dims: int):
        super(_Branch_TextureEmbedding, self).__init__()
        self._0_block = nn.Sequential(
            InceptionV4.Inception_A(),
            InceptionV4.Inception_A(),
            InceptionV4.Inception_A(),
            InceptionV4.Inception_A(),
            InceptionV4.Reduction_A(),
        )

        self._1_block = nn.Sequential(
            InceptionV4.Inception_B(),
            InceptionV4.Inception_B(),
            InceptionV4.Inception_B(),
            InceptionV4.Inception_B(),
            InceptionV4.Inception_B(),
            InceptionV4.Inception_B(),
            InceptionV4.Inception_B(),
            InceptionV4.Reduction_B(),
        )

        self._2_block = nn.Sequential(
            InceptionV4.Inception_C(),
            InceptionV4.Inception_C(),
            InceptionV4.Inception_C(),
        )

        self._3_avg_pool2d = nn.AvgPool2d(
            kernel_size=8
        )  # Might need adjustment if the input size is changed
        self._4_flatten = nn.Flatten()
        self._5_dropout = nn.Dropout(p=0.2)
        self._6_linear = nn.Linear(1536, texture_embedding_dims)

    def forward(self, input):
        x = self._0_block(input)
        x = self._1_block(x)
        x = self._2_block(x)
        x = self._3_avg_pool2d(x)
        x = self._4_flatten(x)
        x = self._5_dropout(x)
        x = self._6_linear(x)
        x = torch.nn.functional.normalize(torch.squeeze(x), dim=1)
        return x


class _Branch_MinutiaStem(nn.Module):
    def __init__(self):
        super().__init__()
        # Modules
        self.features = nn.Sequential(
            InceptionV4.Inception_A(),
            InceptionV4.Inception_A(),
            InceptionV4.Inception_A(),
            InceptionV4.Inception_A(),
            InceptionV4.Inception_A(),
            InceptionV4.Inception_A(),
        )

    def forward(self, input):
        return self.features(input)


class _Branch_MinutiaEmbedding(nn.Module):
    def __init__(self, minutia_embedding_dims: int):
        super().__init__()
        # Modules
        self._0_block = nn.Sequential(
            nn.Conv2d(384, 768, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(768, 896, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(896, 1024, kernel_size=3, stride=2, padding=1),
        )
        self._1_max_pool2d = nn.MaxPool2d(kernel_size=9, stride=1)
        self._2_flatten = nn.Flatten()
        self._3_dropout = nn.Dropout(p=0.2)
        self._4_linear = nn.Linear(1024, minutia_embedding_dims)

    def forward(self, input):
        x = self._0_block(input)
        x = self._1_max_pool2d(x)
        x = self._2_flatten(x)
        x = self._3_dropout(x)
        x = self._4_linear(x)
        x = torch.nn.functional.normalize(x, dim=1)
        return x


class _Branch_MinutiaMap(nn.Module):
    def __init__(self):
        super().__init__()
        # Modules
        self.features = nn.Sequential(
            nn.ConvTranspose2d(384, 128, kernel_size=3, stride=2),
            nn.Conv2d(128, 128, kernel_size=7, stride=1),
            nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2),
            nn.Conv2d(32, 6, kernel_size=3, stride=1),
        )

    def forward(self, input):
        # The network produces maps of size 129x129 but we want maps of size 128x128
        # so we remove the last row / column from each map
        return self.features(input)[:, :, :-1, :-1]


"""
  _ __ ___   ___   __| | ___| | __   ____ _ _ __(_) __ _ _ __ | |_ ___ 
 | '_ ` _ \ / _ \ / _` |/ _ \ | \ \ / / _` | '__| |/ _` | '_ \| __/ __|
 | | | | | | (_) | (_| |  __/ |  \ V / (_| | |  | | (_| | | | | |_\__ \ 
 |_| |_| |_|\___/ \__,_|\___|_|   \_/ \__,_|_|  |_|\__,_|_| |_|\__|___/                                                                    
"""


class DeepPrintOutput:
    def __init__(
        self,
        minutia_embeddings: torch.Tensor = None,
        texture_embeddings: torch.Tensor = None,
    ):
        self.minutia_embeddings: torch.Tensor = minutia_embeddings
        self.texture_embeddings: torch.Tensor = texture_embeddings

    @staticmethod
    def training():
        return False


class DeepPrintTrainingOutput(DeepPrintOutput):
    def __init__(
        self,
        minutia_logits: torch.Tensor = None,
        texture_logits: torch.Tensor = None,
        combined_logits: torch.Tensor = None,
        minutia_maps: torch.Tensor = None,
        **kwargs
    ):
        self.minutia_logits: torch.Tensor = minutia_logits
        self.texture_logits: torch.Tensor = texture_logits
        self.combined_logits: torch.Tensor = combined_logits
        self.minutia_maps: torch.Tensor = minutia_maps
        super().__init__(**kwargs)

    @staticmethod
    def training():
        return True


class DeepPrint_Tex(nn.Module):
    """
    Model with only the texture branch.

    In training mode:
        Outputs the texture embedding AND
        A vector of propabilities over all classes (each class being one subject)
    In evaluation mode:
        Outputs the texture embedding
    """

    def __init__(self, num_fingerprints: int, texture_embedding_dims: int):
        super().__init__()
        # Modules
        self.stem = _InceptionV4_Stem()
        self.texture_branch = _Branch_TextureEmbedding(
            texture_embedding_dims=texture_embedding_dims
        )
        self.texture_logits = nn.Sequential(
            nn.Linear(texture_embedding_dims, num_fingerprints), nn.Dropout(p=0.2)
        )

    def forward(self, input) -> torch.Tensor:
        if self.training:
            x = self.stem(input)
            x = self.texture_branch.forward(x)
            logits = self.texture_logits.forward(x)
            return DeepPrintTrainingOutput(texture_logits=logits, texture_embeddings=x)

        with torch.no_grad():
            x = self.stem(input)
            x = self.texture_branch.forward(x)
            return DeepPrintOutput(texture_embeddings=x)


class DeepPrint_Minu(nn.Module):
    """
    Model with only the minutia branch.

    In training mode:
        Outputs the minutia embedding AND
        A vector of propabilities over all classes (each class being one subject) AND
        Predicted minutia maps
    In evaluation mode:
        Outputs the minutia embeddings
    """

    def __init__(self, num_fingerprints: int, minutia_embedding_dims: int):
        super().__init__()
        # Modules
        self.stem = _InceptionV4_Stem()
        self.minutia_stem = _Branch_MinutiaStem()
        self.minutia_map = _Branch_MinutiaMap()
        self.minutia_embedding = _Branch_MinutiaEmbedding(minutia_embedding_dims)
        self.minutia_logits = nn.Sequential(
            nn.Linear(minutia_embedding_dims, num_fingerprints), nn.Dropout(p=0.2)
        )

    def forward(self, input: torch.Tensor) -> DeepPrintOutput:
        if self.training:
            x = self.stem(input)

            x_minutia = self.minutia_stem.forward(x)
            x_minutia_emb = self.minutia_embedding.forward(x_minutia)
            x_minutia_logits = self.minutia_logits(x_minutia_emb)
            x_minutia_map = self.minutia_map(x_minutia)

            return DeepPrintTrainingOutput(
                minutia_logits=x_minutia_logits,
                minutia_maps=x_minutia_map,
                minutia_embeddings=x_minutia_emb,
            )

        with torch.no_grad():
            x = self.stem(input)
            x_minutia = self.minutia_stem.forward(x)

            x_minutia_emb = self.minutia_embedding.forward(x_minutia)
            return DeepPrintOutput(minutia_embeddings=x_minutia_emb)


class DeepPrint_TexMinu(nn.Module):
    """
    Model with texture and minutia branch

    In training mode:
        Outputs the texture embedding AND
        A vector of propabilities over all classes predicted from the texture embedding
    In evaluation mode:
        Outputs the texture embedding

    """

    def __init__(
        self, num_fingerprints, texture_embedding_dims: int, minutia_embedding_dims: int
    ):
        super().__init__()
        # Modules
        self.stem = _InceptionV4_Stem()
        self.texture_branch = _Branch_TextureEmbedding(texture_embedding_dims)
        self.texture_logits = nn.Sequential(
            nn.Linear(texture_embedding_dims, num_fingerprints), nn.Dropout(p=0.2)
        )
        self.minutia_stem = _Branch_MinutiaStem()
        self.minutia_map = _Branch_MinutiaMap()
        self.minutia_embedding = _Branch_MinutiaEmbedding(minutia_embedding_dims)
        self.minutia_logits = nn.Sequential(
            nn.Linear(minutia_embedding_dims, num_fingerprints), nn.Dropout(p=0.2)
        )

    def forward(self, input: torch.Tensor) -> DeepPrintOutput:
        if self.training:
            x = self.stem(input)

            x_texture_emb = self.texture_branch.forward(x)
            x_texture_logits = self.texture_logits(x_texture_emb)

            x_minutia = self.minutia_stem.forward(x)
            x_minutia_emb = self.minutia_embedding.forward(x_minutia)
            x_minutia_logits = self.minutia_logits(x_minutia_emb)
            x_minutia_map = self.minutia_map(x_minutia)

            return DeepPrintTrainingOutput(
                minutia_logits=x_minutia_logits,
                texture_logits=x_texture_logits,
                minutia_maps=x_minutia_map,
                minutia_embeddings=x_minutia_emb,
                texture_embeddings=x_texture_emb,
            )

        with torch.no_grad():
            x = self.stem(input)
            x_texture_emb = self.texture_branch.forward(x)
            x_minutia = self.minutia_stem.forward(x)

            x_minutia_emb = self.minutia_embedding.forward(x_minutia)
            return DeepPrintOutput(x_minutia_emb, x_texture_emb)


class DeepPrint_LocTex(nn.Module):
    """
    Model with texture branch and localization network.

    In training mode:
        Outputs the texture embedding AND
        A vector of propabilities over all classes predicted from the texture embedding
    In evaluation mode:
        Outputs the texture embedding

    """

    def __init__(self, num_fingerprints, texture_embedding_dims: int):
        super().__init__()
        # Special attributs
        self.input_space = None

        # Modules
        self.localization = LocalizationNetwork()
        self.embeddings = DeepPrint_Tex(
            num_fingerprints=num_fingerprints,
            texture_embedding_dims=texture_embedding_dims,
        )

    def forward(self, input: torch.Tensor) -> DeepPrintOutput:
        if self.training:
            x = self.localization(input)
            self.embeddings.train()
            return self.embeddings(x)

        with torch.no_grad():
            x = self.localization(input)
            self.embeddings.eval()
            return self.embeddings(x)


class DeepPrint_LocMinu(nn.Module):
    """
    Model with minutia branch and localization network.

    In training mode:
        Outputs the minutia embedding AND
        A vector of propabilities over all classes predicted from the minutia embedding AND
        The generated minutia maps
    In evaluation mode:
        Outputs the minutia embedding

    """

    def __init__(self, num_fingerprints, minutia_embedding_dims: int):
        super().__init__()
        # Special attributs
        self.input_space = None

        # Modules
        self.localization = LocalizationNetwork()
        self.embeddings = DeepPrint_Minu(
            num_fingerprints=num_fingerprints,
            minutia_embedding_dims=minutia_embedding_dims,
        )

    def forward(self, input: torch.Tensor) -> DeepPrintOutput:
        if self.training:
            x = self.localization(input)
            self.embeddings.train()
            return self.embeddings(x)

        with torch.no_grad():
            x = self.localization(input)
            self.embeddings.eval()
            return self.embeddings(x)


class DeepPrint_LocTexMinu(nn.Module):
    """
    Model with texture and minutia branch and localization network.

    In training mode:
        Outputs the texture embedding AND
        A vector of propabilities over all classes predicted from the texture embedding
    In evaluation mode:
        Outputs the texture embedding

    """

    def __init__(
        self, num_fingerprints, texture_embedding_dims: int, minutia_embedding_dims: int
    ):
        super().__init__()
        # Special attributs
        self.input_space = None

        # Modules
        self.localization = LocalizationNetwork()
        self.embeddings = DeepPrint_TexMinu(
            num_fingerprints=num_fingerprints,
            texture_embedding_dims=texture_embedding_dims,
            minutia_embedding_dims=minutia_embedding_dims,
        )

    def forward(self, input: torch.Tensor) -> DeepPrintOutput:
        if self.training:
            x = self.localization(input)
            self.embeddings.train()
            return self.embeddings(x)

        with torch.no_grad():
            x = self.localization(input)
            self.embeddings.eval()
            return self.embeddings(x)


def main():
    model = DeepPrint_Tex(1000, 100)
    model = DeepPrint_Minu(1000, 100)
    model = DeepPrint_TexMinu(1000, 100, 100)
    model = DeepPrint_LocTex(1000, 100)
    model = DeepPrint_LocMinu(1000, 100)
    model = DeepPrint_LocTexMinu(1000, 100, 100)
    print("no syntax errors")


if __name__ == "__main__":
    main()
