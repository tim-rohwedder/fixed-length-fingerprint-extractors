from typing import Union
from dataclasses import dataclass
from os.path import join
from enum import Enum

from flx.benchmarks.verification import VerificationResult
from flx.benchmarks.identification import IdentificationResult
from flx.setup.experiments import (
    get_experiments,
    get_reweighting_experiments,
    Experiment,
    DatasetLoader,
)
from flx.setup.paths import get_figures_dir
from flx.visualization.plot_DET_curve import (
    plot_verification_results,
    plot_identification_results,
)
from flx.visualization.plot_ranks import plot_rank_n_identification_rates
from flx.visualization.plot_distribution_scores import plot_similarity_scores_results


class EmbeddingType(Enum):
    Original = 0
    Reweighted = 1
    Both = 2


@dataclass
class PlotConfig:
    figures_subdir: str
    extractor_keys: list[str]
    label: str
    plot_distributions: bool = False
    plot_verification: bool = False
    plot_closed_set: bool = False
    plot_open_set: bool = False
    embedding_types: Union[EmbeddingType, list[EmbeddingType]] = EmbeddingType.Original


PLOT_CONFIGS: list[PlotConfig] = [
    PlotConfig(
        "synthetic_vs_mixed",
        [
            "DeepPrint_Texture_first4000",
            "DeepPrint_Texture_Mixed_contrast",
        ],
        "Purely synthetic training set compared to\ntraining set with real and synthetic data",
        plot_verification=True,
    ),
    PlotConfig(
        "preprocessing",
        [
            "DeepPrint_Texture_Mixed_contrast",
            "DeepPrint_Texture_Mixed_gabor",
            "DeepPrint_Texture_Mixed_gabor3",
        ],
        "Models with different preprocessing approaches",
        plot_verification=True,
    ),
    PlotConfig(
        "reweighting",
        [
            "DeepPrint_Tex_512",
        ],
        "Different variants",
        embedding_types=EmbeddingType.Both,
        plot_verification=True,
        plot_distributions=True,
    ),
    PlotConfig(
        "embedding_sizes",
        [
            "DeepPrint_Tex_32",
            "DeepPrint_Tex_64",
            "DeepPrint_Tex_128",
            "DeepPrint_Tex_256",
            "DeepPrint_Tex_512",
            "DeepPrint_Tex_1024",
            "DeepPrint_Tex_2048",
        ],
        "Different embedding sizes",
        plot_verification=True,
    ),
    PlotConfig(
        "embedding_sizes_rw",
        [
            "DeepPrint_Tex_32",
            "DeepPrint_Tex_64",
            "DeepPrint_Tex_128",
            "DeepPrint_Tex_256",
            "DeepPrint_Tex_512",
            "DeepPrint_Tex_1024",
            "DeepPrint_Tex_2048",
        ],
        "Different embedding sizes (reweighted)",
        embedding_types=EmbeddingType.Reweighted,
        plot_verification=True,
    ),
    PlotConfig(
        "variants",
        [
            "DeepPrint_Tex_512",
            "DeepPrint_Minu_512",
            "DeepPrint_TexMinu_512",
        ],
        "Different variants",
        embedding_types=[
            EmbeddingType.Reweighted,
            EmbeddingType.Original,
            EmbeddingType.Original,
        ],
        plot_verification=True,
        plot_distributions=False,
        plot_open_set=False,
        plot_closed_set=True,
    ),
]
PLOT_CONFIGS = PLOT_CONFIGS[-1:]


TESTSET_KEYS = [
    "mcyt330_optical",
    "mcyt330_capacitive",
]

MAKE_TITLE = False
SKIP_DISTRIBUTIONS = False
SKIP_VERIFICATION = False
SKIP_CLOSED_SET = False
SKIP_OPEN_SET = False


def _plot_results_on_dataset(
    config: PlotConfig, dataset: DatasetLoader, experiments: list[Experiment]
) -> None:
    experiments_ds: list[Experiment] = [
        e for e in experiments if e.dataset_name == dataset.name
    ]
    plotdir = get_figures_dir(config.figures_subdir, dataset.name)
    model_labels = [e.model_label for e in experiments_ds]

    if config.plot_distributions and not SKIP_DISTRIBUTIONS:
        print(f"Plotting distributions on {dataset.name}")
        results: list[VerificationResult] = [
            e.load_verification_benchmark_results() for e in experiments_ds
        ]
        plot_similarity_scores_results(
            paths=[join(plotdir, f"{e.model_name}.png") for e in experiments_ds],
            results=results,
            plot_titles=[
                f"Distribution of mated and non-mated similarities\nfor {m}\non {dataset.label}"
                for m in model_labels
            ]
            if MAKE_TITLE
            else [""] * len(model_labels),
        )

    if config.plot_verification and not SKIP_VERIFICATION:
        print(f"Plotting FMR-FNMR of {dataset.name}")
        results: list[VerificationResult] = [
            e.load_verification_benchmark_results() for e in experiments_ds
        ]
        plot_verification_results(
            join(plotdir, "verification"),
            results=results,
            model_labels=model_labels,
            plot_title=f"Verification performance on {dataset.label}"
            if MAKE_TITLE
            else "",
        )

    if config.plot_closed_set and not SKIP_CLOSED_SET:
        print(f"Plotting rank-N identification rates of {dataset.name}")
        results: list[IdentificationResult] = [
            e.load_closed_set_benchmark_results() for e in experiments_ds
        ]
        plot_rank_n_identification_rates(
            join(plotdir, "identification_closed_set"),
            results=results,
            model_labels=model_labels,
            plot_title=f"Rank-N identification rates on {dataset.label}"
            if MAKE_TITLE
            else "",
        )

    if config.plot_open_set and not SKIP_OPEN_SET:
        print(f"Plotting FPIR-FNIR of {dataset.name}")
        results: list[IdentificationResult] = [
            e.load_open_set_benchmark_results() for e in experiments_ds
        ]
        plot_identification_results(
            join(plotdir, "identification_open_set"),
            results=results,
            model_labels=model_labels,
            plot_title=f"Identification performance on {dataset.label}"
            if MAKE_TITLE
            else "",
        )


def _get_experiments_with_rw(
    testset_keys: list[str],
    extractor_keys: list[str],
    embedding_types: Union[EmbeddingType, list[EmbeddingType]],
):
    if isinstance(embedding_types, EmbeddingType):
        embedding_types = [embedding_types for _ in extractor_keys]
    embedding_types = {k: t for k, t in zip(extractor_keys, embedding_types)}

    testsets, _, experiments = get_experiments(
        testset_keys=testset_keys, extractor_keys=extractor_keys
    )
    orig_experiments = {
        k: e
        for k, e in experiments.items()
        if embedding_types[k[0]] == EmbeddingType.Original
        or embedding_types[k[0]] == EmbeddingType.Both
    }
    rw_experiments = get_reweighting_experiments(
        {
            k: e
            for k, e in experiments.items()
            if embedding_types[k[0]] == EmbeddingType.Reweighted
            or embedding_types[k[0]] == EmbeddingType.Both
        }
    )

    return testsets, list(orig_experiments.values()) + list(rw_experiments.values())


def main():
    for plot_config in PLOT_CONFIGS:
        datasets, experiments_lst = _get_experiments_with_rw(
            TESTSET_KEYS, plot_config.extractor_keys, plot_config.embedding_types
        )

        for dataset in datasets:
            _plot_results_on_dataset(plot_config, dataset, experiments_lst)


if __name__ == "__main__":
    main()
