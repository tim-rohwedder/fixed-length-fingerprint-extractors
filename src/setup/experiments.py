from typing import Callable, Union
from dataclasses import dataclass

import os

from src.setup.paths import (
    get_model_dir,
    get_fingerprint_dataset_path,
    get_verification_benchmark_file,
)
from src.setup._experiment import Experiment, ReweightingExperiment

from src.benchmarks.verification import VerificationBenchmark
from src.data.biometric_dataset import BiometricDataset
from src.setup.datasets import (
    get_sfinge_example,
    get_sfinge_validation_separate_subjects,
    get_sfinge_test_none,
    get_sfinge_test_optical,
    get_sfinge_test_capacitive,
    get_fvc2004_db1a,
    get_mcyt_optical,
    get_mcyt_capacitive,
    get_nist_sd4,
    get_nist_sd14,
)
from src.extractor.fixed_length_extractor import (
    FixedLengthExtractor,
    get_DeepFinger_Tex,
    get_DeepFinger_Minu,
    get_DeepFinger_TexMinu,
    get_DeepFinger_TexMinuCombi,
    get_DeepFinger_LocTex,
    get_DeepFinger_LocMinu,
    get_DeepFinger_LocTexMinu,
    get_ViTB32_Pretrained,
)


@dataclass
class FixedLengthExtractorLoader:
    name: str
    label: str
    constructor: Callable[[], FixedLengthExtractor]

    def load(self) -> FixedLengthExtractor:
        extractor = self.constructor()
        model_dir = get_model_dir(self.name)
        extractor.load_best_model(model_dir)
        return extractor

    def get_dir(self) -> str:
        return get_model_dir(self.name)


@dataclass
class DatasetLoader:
    name: str
    label: str
    constructor: Callable[[str], BiometricDataset]
    indices_in_training_set: list[int] = None
    kwargs: dict = None

    def load(self) -> BiometricDataset:
        if self.kwargs is not None:
            return self.constructor(get_fingerprint_dataset_path(self.name), **self.kwargs)
        return self.constructor(get_fingerprint_dataset_path(self.name))

    def load_verification_benchmark(self) -> VerificationBenchmark:
        return VerificationBenchmark.load(get_verification_benchmark_file(self.name))


@dataclass
class BenchmarkLoader:
    model_name: str
    model_label: str


EXTRACTORS = {
    loader.name: loader
    for loader in [
        # To compare different training sets
        FixedLengthExtractorLoader(
            "DeepFinger_Texture_first4000",
            "DeepPrint (texture branch)\nTrained on the first 4000 subjects of SFingev2",
            lambda: get_DeepFinger_Tex(4000, 96),
        ),
        FixedLengthExtractorLoader(
            "DeepFinger_Texture_Mixed",
            "DeepPrint (texture branch)\nTrained on mixed dataset",
            lambda: get_DeepFinger_Tex(2800 + 8000, 96),
        ),
        # The following models are for comparing different preprocessing techniques
        FixedLengthExtractorLoader(
            "DeepFinger_Texture_Mixed_gabor",
            "DeepPrint (texture branch, Gabor filter strong smoothing)",
            lambda: get_DeepFinger_Tex(2800 + 4000, 96),
        ),
        FixedLengthExtractorLoader(
            "DeepFinger_Texture_Mixed_gabor3",
            "DeepPrint (texture branch, Gabor filter less smoothing)",
            lambda: get_DeepFinger_Tex(2800 + 4000, 96),
        ),
        FixedLengthExtractorLoader(
            "DeepFinger_Texture_Mixed_contrast",
            "DeepPrint (texture branch, contrast enhancement)",
            lambda: get_DeepFinger_Tex(2800 + 4000, 96),
        ),
        # Different embedding sizes
        FixedLengthExtractorLoader(
            "DeepFinger_Tex_32",
            "DeepPrint (texture branch, 32 dims)",
            lambda: get_DeepFinger_Tex(2000 + 6000, 32),
        ),
        FixedLengthExtractorLoader(
            "DeepFinger_Tex_64",
            "DeepPrint (texture branch, 64 dims)",
            lambda: get_DeepFinger_Tex(2000 + 6000, 64),
        ),
        FixedLengthExtractorLoader(
            "DeepFinger_Tex_128",
            "DeepPrint (texture branch, 128 dims)",
            lambda: get_DeepFinger_Tex(2000 + 6000, 128),
        ),
        FixedLengthExtractorLoader(
            "DeepFinger_Tex_256",
            "DeepPrint (texture branch, 256 dims)",
            lambda: get_DeepFinger_Tex(2000 + 6000, 256),
        ),
        FixedLengthExtractorLoader(
            "DeepFinger_Tex_512",
            "DeepPrint (texture branch, 512 dims)",
            lambda: get_DeepFinger_Tex(2000 + 6000, 512),
        ),
        FixedLengthExtractorLoader(
            "DeepFinger_Tex_1024",
            "DeepPrint (texture branch, 1024 dims)",
            lambda: get_DeepFinger_Tex(2000 + 6000, 1024),
        ),
        FixedLengthExtractorLoader(
            "DeepFinger_Tex_2048",
            "DeepPrint (texture branch, 2048 dims)",
            lambda: get_DeepFinger_Tex(2000 + 6000, 2048),
        ),
        FixedLengthExtractorLoader(
            "DeepPrintMCYT_TexMinu_512",
            "DeepPrint (mcyt, texture and minutia branch, 512 dims)",
            lambda: get_DeepFinger_TexMinu(2000, 512),
        ),
        # Different variants
        FixedLengthExtractorLoader(
            "DeepFinger_Minu_512",
            "DeepPrint (minutia branch, 512 dims)",
            lambda: get_DeepFinger_Minu(2000 + 6000, 512),
        ),
        FixedLengthExtractorLoader(
            "DeepFinger_TexMinu_512",
            "DeepPrint (texture and minutia branch, 512 dims)",
            lambda: get_DeepFinger_TexMinu(2000 + 6000, 256),
        ),
        FixedLengthExtractorLoader(
            "DeepFinger_TexMinuCombi_512",
            "DeepPrint (texture and minutia branch with combined logits, 512 dims)",
            lambda: get_DeepFinger_TexMinuCombi(2000 + 6000, 256),
        ),
        FixedLengthExtractorLoader(
            "DeepFinger_512",
            "DeepPrint (full architecture, separate logits for both branches)",
            lambda: get_DeepFinger_LocTexMinu(2000 + 6000, 256),
        ),
        FixedLengthExtractorLoader(
            "DeepFinger_LocTex_512",
            "DeepPrint (localization module and texture branch)",
            lambda: get_DeepFinger_LocTex(2000 + 6000, 512),
        ),
        FixedLengthExtractorLoader(
            "DeepFinger_LocMinu_512",
            "DeepPrint (localization module and minutia branch)",
            lambda: get_DeepFinger_LocMinu(2000 + 6000, 512),
        ),
        FixedLengthExtractorLoader(
            "ViTB32_Pretrained_512",
            "ViT B32 Pretrained (512 dims)",
            lambda: get_ViTB32_Pretrained(2000 + 6000, 512),
        ),
    ]
}

TESTSETS = {
    loader.name: loader
    for loader in [
        DatasetLoader(
            "SFingev2Example",
            "SFinge (small example)",
            get_sfinge_example,
        ),
        DatasetLoader(
            "SFingev2ValidationSeparateSubjects",
            "SFinge (separate subjects and samples from the training set\nused for validation)",
            get_sfinge_validation_separate_subjects,
        ),
        DatasetLoader(
            "SFingev2TestNone",
            "SFinge testset (white background)",
            get_sfinge_test_none,
        ),
        DatasetLoader(
            "SFingev2TestCapacitive",
            "SFinge testset (capactive sensor)",
            get_sfinge_test_capacitive,
        ),
        DatasetLoader(
            "SFingev2TestOptical",
            "SFinge testset (optical sensor)",
            get_sfinge_test_optical,
        ),
        DatasetLoader(
            "FVC2004_DB1A",
            "FVC2004 DB1 A",
            get_fvc2004_db1a,
        ),
        DatasetLoader(
            "NIST SD4",
            "NIST SD4",
            get_nist_sd4,
        ),
        DatasetLoader(
            "NIST SD14",
            "NIST SD14 (last 2700 pairs)",
            get_nist_sd14,
            list(range(24300 * 2)),
        ),
        DatasetLoader(
            "mcyt330_optical",
            "MCYT330 (optical sensor, last 1300 subjects)",
            get_mcyt_optical,
            list(range(2000 * 12)),
        ),
        DatasetLoader(
            "mcyt330_capacitive",
            "MCYT330 (capacitive sensor, last 1300 subjects)",
            get_mcyt_capacitive,
            list(range(2000 * 12)),
        ),
    ]
}


def get_experiments(
    extractor_keys: Union[list[str], None] = None,
    testset_keys: Union[list[str], None] = None,
) -> tuple[
    list[DatasetLoader],
    list[FixedLengthExtractorLoader],
    dict[tuple[str, str], Experiment],
]:
    testsets = (
        TESTSETS.values()
        if testset_keys is None
        else [TESTSETS[k] for k in testset_keys]
    )
    extractors = (
        EXTRACTORS.values()
        if extractor_keys is None
        else [EXTRACTORS[k] for k in extractor_keys]
    )
    experiments = {
        (e.name, d.name): Experiment(e.name, e.label, d.name, d.label, reweighting_training_indices=d.indices_in_training_set)
        for e in extractors
        for d in testsets
    }
    return testsets, extractors, experiments


def get_reweighting_experiments(experiments: dict[tuple[str, str], Experiment]) -> dict[tuple[str, str], ReweightingExperiment]:
    return {k: ReweightingExperiment(
        model_name=e.model_name,
        model_label=e.model_label,
        dataset_name=e.dataset_name,
        dataset_label=e.dataset_label,
        reweighting_training_indices=e.reweighting_training_indices
        ) for k, e in experiments.items()}
