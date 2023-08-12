from typing import Callable, Union
from dataclasses import dataclass

import os

from flx.setup.paths import (
    get_model_dir,
    get_fingerprint_dataset_path,
    get_verification_benchmark_file,
)
from flx.setup._experiment import Experiment, ReweightingExperiment

from flx.benchmarks.verification import VerificationBenchmark
from flx.data.dataset import Dataset
from flx.setup.datasets import (
    get_sfinge_example,
    get_sfinge_validation_separate_subjects,
    get_sfinge_test,
    get_fvc2004_db1a,
    get_mcyt_optical,
    get_mcyt_capacitive,
    get_nist_sd4,
)
from flx.extractor.fixed_length_extractor import (
    DeepPrintExtractor,
    get_DeepPrint_Tex,
    get_DeepPrint_Minu,
    get_DeepPrint_TexMinu,
    get_DeepPrint_LocTex,
    get_DeepPrint_LocMinu,
    get_DeepPrint_LocTexMinu,
)


@dataclass
class FixedLengthExtractorLoader:
    name: str
    label: str
    constructor: Callable[[], DeepPrintExtractor]

    def load(self) -> DeepPrintExtractor:
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
    constructor: Callable[[str], Dataset]

    def load(self) -> Dataset:
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
        # Different embedding sizes
        FixedLengthExtractorLoader(
            "DeepPrint_Tex_32",
            "DeepPrint (texture branch, 32 dims)",
            lambda: get_DeepPrint_Tex(2000 + 6000, 32),
        ),
        FixedLengthExtractorLoader(
            "DeepPrint_Tex_64",
            "DeepPrint (texture branch, 64 dims)",
            lambda: get_DeepPrint_Tex(2000 + 6000, 64),
        ),
        FixedLengthExtractorLoader(
            "DeepPrint_Tex_128",
            "DeepPrint (texture branch, 128 dims)",
            lambda: get_DeepPrint_Tex(2000 + 6000, 128),
        ),
        FixedLengthExtractorLoader(
            "DeepPrint_Tex_256",
            "DeepPrint (texture branch, 256 dims)",
            lambda: get_DeepPrint_Tex(2000 + 6000, 256),
        ),
        FixedLengthExtractorLoader(
            "DeepPrint_Tex_512",
            "DeepPrint (texture branch, 512 dims)",
            lambda: get_DeepPrint_Tex(2000 + 6000, 512),
        ),
        FixedLengthExtractorLoader(
            "DeepPrint_Tex_1024",
            "DeepPrint (texture branch, 1024 dims)",
            lambda: get_DeepPrint_Tex(2000 + 6000, 1024),
        ),
        FixedLengthExtractorLoader(
            "DeepPrint_Tex_2048",
            "DeepPrint (texture branch, 2048 dims)",
            lambda: get_DeepPrint_Tex(2000 + 6000, 2048),
        ),
        # Different variants
        FixedLengthExtractorLoader(
            "DeepPrint_Minu_512",
            "DeepPrint (minutia branch, 512 dims)",
            lambda: get_DeepPrint_Minu(2000 + 6000, 512),
        ),
        FixedLengthExtractorLoader(
            "DeepPrint_TexMinu_512",
            "DeepPrint (texture and minutia branch, 512 dims)",
            lambda: get_DeepPrint_TexMinu(2000 + 6000, 256),
        ),
        FixedLengthExtractorLoader(
            "DeepPrint_LocTex_512",
            "DeepPrint (localization module and texture branch)",
            lambda: get_DeepPrint_LocTex(2000 + 6000, 512),
        ),
        FixedLengthExtractorLoader(
            "DeepPrint_LocMinu_512",
            "DeepPrint (localization module and minutia branch)",
            lambda: get_DeepPrint_LocMinu(2000 + 6000, 512),
        ),
        FixedLengthExtractorLoader(
            "DeepPrint_LocTexMinu_512",
            "DeepPrint (full architecture, separate logits for both branches)",
            lambda: get_DeepPrint_LocTexMinu(2000 + 6000, 256),
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
            get_sfinge_test,
        ),
        DatasetLoader(
            "SFingev2TestCapacitive",
            "SFinge testset (capactive sensor)",
            get_sfinge_test,
        ),
        DatasetLoader(
            "SFingev2TestOptical",
            "SFinge testset (optical sensor)",
            get_sfinge_test,
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
            "mcyt330_optical",
            "MCYT330 (optical sensor, last 1300 subjects)",
            lambda x: get_mcyt_optical(x, poses=None, only_last_n=1300),
        ),
        DatasetLoader(
            "mcyt330_capacitive",
            "MCYT330 (capacitive sensor, last 1300 subjects)",  
            lambda x: get_mcyt_capacitive(x, poses=None, only_last_n=1300),
        ),
    ]
}

REWEIGHTING_TRAINING_INDICES = {
    "mcyt330_optical": list(range(2000 * 12)),
    "mcyt330_capacitive": list(range(2000 * 12)),
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
        (e.name, d.name): Experiment(
            e.name,
            e.label,
            d.name,
            d.label,
            reweighting_training_indices=REWEIGHTING_TRAINING_INDICES.get(d.name),
        )
        for e in extractors
        for d in testsets
    }
    return testsets, extractors, experiments


def get_reweighting_experiments(
    experiments: dict[tuple[str, str], Experiment]
) -> dict[tuple[str, str], ReweightingExperiment]:
    return {
        k: ReweightingExperiment(
            model_name=e.model_name,
            model_label=e.model_label,
            dataset_name=e.dataset_name,
            dataset_label=e.dataset_label,
            reweighting_training_indices=e.reweighting_training_indices,
        )
        for k, e in experiments.items()
    }
