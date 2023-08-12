from dataclasses import dataclass
from os.path import exists

import torch
from flx.models.HyPrint import ViTB32Pretrained

from flx.benchmarks.verification import VerificationBenchmark
from flx.data.embedding_dataset import EmbeddingDataset
from flx.extractor.extract_embeddings import extract_embeddings
from flx.extractor.model_training import train_model
from flx.models.DeepFinger import (
    DeepFinger_Tex,
    DeepFinger_LocTexMinu,
    DeepFinger_TexMinu,
    DeepFinger_TexMinuCombi,
    DeepFinger_LocTex,
    DeepFinger_LocMinu,
    DeepFinger_Minu,
)
from flx.models.DeepFinger_loss import (
    DeepFingerLoss_Tex,
    DeepFingerLoss_Minu,
    DeepFingerLoss_TexMinu,
    DeepFingerLoss_TexMinuCombi
)
from flx.setup.paths import get_best_model_file
from flx.data.biometric_dataset import BiometricDataset
from flx.utils.torch_helpers import load_model_parameters, get_device
from flx.data.biometric_dataset import DummyDataset


@dataclass
class FixedLengthExtractor:
    training_with_minutia_map: bool
    model: torch.nn.Module
    loss: torch.nn.Module

    def fit(
        self,
        fingerprints: BiometricDataset,
        minutia_maps: BiometricDataset,
        labels: BiometricDataset,
        validation_fingerprints: BiometricDataset,
        validation_benchmark: VerificationBenchmark,
        num_epochs: int,
        out_dir: str,
    ) -> None:
        if not self.training_with_minutia_map:
            minutia_maps = DummyDataset(minutia_maps.ids, (torch.tensor([]), 0.0))
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

    def predict(
        self, dataset: BiometricDataset
    ) -> tuple[EmbeddingDataset, EmbeddingDataset]:
        return extract_embeddings(self.model, dataset)

    def load_best_model(self, model_dir: str) -> None:
        model_path = get_best_model_file(model_dir)
        if exists(model_path):
            print(f"Loaded best model from {model_path}")
            load_model_parameters(model_path, self.model, None, None)
        else:
            print(f"No best model file found at {model_path}")


def get_DeepFinger_Tex(
    num_training_subjects: int, num_texture_dims: int
) -> FixedLengthExtractor:
    model = DeepFinger_Tex(num_training_subjects, num_texture_dims)
    loss = DeepFingerLoss_Tex(num_training_subjects, num_texture_dims)
    return FixedLengthExtractor(
        training_with_minutia_map=False,
        model=model,
        loss=loss,
    )


def get_DeepFinger_Minu(
    num_training_subjects: int, num_minutia_dims: int
) -> FixedLengthExtractor:
    model = DeepFinger_Minu(num_training_subjects, num_minutia_dims)
    loss = DeepFingerLoss_Minu(num_training_subjects, num_minutia_dims)
    return FixedLengthExtractor(
        training_with_minutia_map=True,
        model=model,
        loss=loss,
    )


def get_DeepFinger_LocTex(
    num_training_subjects: int, num_texture_dims: int
) -> FixedLengthExtractor:
    model = DeepFinger_LocTex(num_training_subjects, num_texture_dims)
    loss = DeepFingerLoss_Tex(num_training_subjects, num_texture_dims)
    optimizer = torch.optim.Adam(params=model.parameters())
    return FixedLengthExtractor(
        training_with_minutia_map=False,
        model=model,
        loss=loss,
    )


def get_DeepFinger_TexMinu(
    num_training_subjects: int, num_dims: int
) -> FixedLengthExtractor:
    model = DeepFinger_TexMinu(num_training_subjects, num_dims, num_dims)
    loss = DeepFingerLoss_TexMinu(num_training_subjects, num_dims, num_dims)
    return FixedLengthExtractor(
        training_with_minutia_map=True,
        model=model,
        loss=loss,
    )


def get_DeepFinger_TexMinuCombi(
    num_training_subjects: int, num_dims: int
) -> FixedLengthExtractor:
    model = DeepFinger_TexMinuCombi(num_training_subjects, num_dims, num_dims)
    loss = DeepFingerLoss_TexMinuCombi(num_training_subjects, num_dims, num_dims)
    return FixedLengthExtractor(
        training_with_minutia_map=True,
        model=model,
        loss=loss,
    )


def get_DeepFinger_LocTex(
    num_training_subjects: int, num_texture_dims: int
) -> FixedLengthExtractor:
    model = DeepFinger_LocTex(num_training_subjects, num_texture_dims)
    loss = DeepFingerLoss_Tex(num_training_subjects, num_texture_dims)
    return FixedLengthExtractor(
        training_with_minutia_map=False,
        model=model,
        loss=loss,
    )


def get_DeepFinger_LocMinu(
    num_training_subjects: int, num_texture_dims: int
) -> FixedLengthExtractor:
    model = DeepFinger_LocMinu(num_training_subjects, num_texture_dims)
    loss = DeepFingerLoss_Minu(num_training_subjects, num_texture_dims)
    return FixedLengthExtractor(
        training_with_minutia_map=True,
        model=model,
        loss=loss,
    )


def get_DeepFinger_LocTexMinu(
    num_training_subjects: int, num_dims: int
) -> FixedLengthExtractor:
    model = DeepFinger_LocTexMinu(num_training_subjects, num_dims, num_dims)
    loss = DeepFingerLoss_TexMinu(num_training_subjects, num_dims, num_dims)
    return FixedLengthExtractor(
        training_with_minutia_map=True,
        model=model,
        loss=loss,
    )


def get_ViTB32_Pretrained(
    num_training_subjects: int, num_dims: int
) -> FixedLengthExtractor:
    return FixedLengthExtractor(
        training_with_minutia_map=False,
        model=ViTB32Pretrained(
            num_classes=num_training_subjects, representation_size=num_dims),
        loss=DeepFingerLoss_Tex(num_training_subjects, num_dims),
    )