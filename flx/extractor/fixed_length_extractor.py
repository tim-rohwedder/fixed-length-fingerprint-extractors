from dataclasses import dataclass
from os.path import exists

import torch

from flx.benchmarks.verification import VerificationBenchmark
from flx.data.dataset import IdentifierSet
from flx.data.embedding_loader import EmbeddingLoader
from flx.data.image_loader import ImageLoader
from flx.extractor.extract_embeddings import extract_embeddings
from flx.models.model_training import train_model
from flx.models.deep_print_arch import (
    DeepPrint_Tex,
    DeepPrint_LocTexMinu,
    DeepPrint_TexMinu,
    DeepPrint_LocTex,
    DeepPrint_LocMinu,
    DeepPrint_Minu,
)
from flx.models.deep_print_loss import (
    DeepPrintLoss_Tex,
    DeepPrintLoss_Minu,
    DeepPrintLoss_TexMinu,
)
from flx.setup.paths import get_best_model_file
from flx.data.dataset import Dataset
from flx.models.torch_helpers import load_model_parameters
from flx.data.dataset import ConstantDataLoader


@dataclass
class DeepPrintExtractor:
    training_with_minutia_map: bool
    model: torch.nn.Module
    loss: torch.nn.Module

    def fit(
        self,
        fingerprints: Dataset,
        minutia_maps: Dataset,
        labels: Dataset,
        validation_fingerprints: Dataset,
        validation_benchmark: VerificationBenchmark,
        num_epochs: int,
        out_dir: str,
    ) -> None:
        if not self.training_with_minutia_map:
            minutia_maps = Dataset(ConstantDataLoader((torch.tensor([]), 0.0)), fingerprints.ids)
        train_model(
            model=self.model,
            loss=self.loss,
            fingerprints=fingerprints,
            minutia_maps=minutia_maps,
            labels=labels,
            validation_fingerprints=validation_fingerprints,
            validation_benchmark=validation_benchmark,
            num_epochs=num_epochs,
            out_dir=out_dir,
        )

    def extract(self, dataset: Dataset) -> tuple[EmbeddingLoader, EmbeddingLoader]:
        return extract_embeddings(self.model, dataset)

    def load_best_model(self, model_dir: str) -> None:
        model_path = get_best_model_file(model_dir)
        if exists(model_path):
            print(f"Loaded best model from {model_path}")
            load_model_parameters(model_path, self.model, None, None)
        else:
            print(f"No best model file found at {model_path}")


def get_DeepPrint_Tex(
    num_training_subjects: int, num_texture_dims: int
) -> DeepPrintExtractor:
    model = DeepPrint_Tex(num_training_subjects, num_texture_dims)
    loss = DeepPrintLoss_Tex(num_training_subjects, num_texture_dims)
    return DeepPrintExtractor(
        training_with_minutia_map=False,
        model=model,
        loss=loss,
    )


def get_DeepPrint_Minu(
    num_training_subjects: int, num_minutia_dims: int
) -> DeepPrintExtractor:
    model = DeepPrint_Minu(num_training_subjects, num_minutia_dims)
    loss = DeepPrintLoss_Minu(num_training_subjects, num_minutia_dims)
    return DeepPrintExtractor(
        training_with_minutia_map=True,
        model=model,
        loss=loss,
    )


def get_DeepPrint_LocTex(
    num_training_subjects: int, num_texture_dims: int
) -> DeepPrintExtractor:
    model = DeepPrint_LocTex(num_training_subjects, num_texture_dims)
    loss = DeepPrintLoss_Tex(num_training_subjects, num_texture_dims)
    optimizer = torch.optim.Adam(params=model.parameters())
    return DeepPrintExtractor(
        training_with_minutia_map=False,
        model=model,
        loss=loss,
    )


def get_DeepPrint_TexMinu(
    num_training_subjects: int, num_dims: int
) -> DeepPrintExtractor:
    model = DeepPrint_TexMinu(num_training_subjects, num_dims, num_dims)
    loss = DeepPrintLoss_TexMinu(num_training_subjects, num_dims, num_dims)
    return DeepPrintExtractor(
        training_with_minutia_map=True,
        model=model,
        loss=loss,
    )


def get_DeepPrint_LocTex(
    num_training_subjects: int, num_texture_dims: int
) -> DeepPrintExtractor:
    model = DeepPrint_LocTex(num_training_subjects, num_texture_dims)
    loss = DeepPrintLoss_Tex(num_training_subjects, num_texture_dims)
    return DeepPrintExtractor(
        training_with_minutia_map=False,
        model=model,
        loss=loss,
    )


def get_DeepPrint_LocMinu(
    num_training_subjects: int, num_texture_dims: int
) -> DeepPrintExtractor:
    model = DeepPrint_LocMinu(num_training_subjects, num_texture_dims)
    loss = DeepPrintLoss_Minu(num_training_subjects, num_texture_dims)
    return DeepPrintExtractor(
        training_with_minutia_map=True,
        model=model,
        loss=loss,
    )


def get_DeepPrint_LocTexMinu(
    num_training_subjects: int, num_dims: int
) -> DeepPrintExtractor:
    model = DeepPrint_LocTexMinu(num_training_subjects, num_dims, num_dims)
    loss = DeepPrintLoss_TexMinu(num_training_subjects, num_dims, num_dims)
    return DeepPrintExtractor(
        training_with_minutia_map=True,
        model=model,
        loss=loss,
    )
