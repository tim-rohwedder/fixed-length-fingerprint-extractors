from os.path import join

import numpy as np

from flx.benchmarks.verification import VerificationResult
from flx.benchmarks.identification import IdentificationResult
from flx.setup.experiments import (
    get_experiments,
    get_reweighting_experiments,
    Experiment,
    DatasetLoader,
)
from flx.generate_embeddings_alignment import get_alignment_dataset_name
from flx.visualization.plot_heatmap import plot_heatmap
from flx.setup.paths import get_figures_dir

import random


def _get_experiment_with_rw(
    testset_key: str, extractor_key: str, rotation: int, shift: int
) -> list[Experiment]:
    _, _, experiments = get_experiments(
        testset_keys=[testset_key], extractor_keys=[extractor_key]
    )
    rw_experiments = get_reweighting_experiments(experiments)

    experiment: Experiment = list(experiments.values())[0]
    experiment.dataset_name = get_alignment_dataset_name(testset_key, rotation, shift)

    rw_experiment: Experiment = list(rw_experiments.values())[0]
    rw_experiment.dataset_name = get_alignment_dataset_name(
        testset_key, rotation, shift
    )

    return experiment, rw_experiment


def _get_fnmr_at_fmr_in_percent(exp: Experiment, fmr: float) -> float:
    return random.random()
    result = exp.load_verification_benchmark_results()
    t = result.threshold_for_fmr(fmr / 100)
    return result.false_non_match_rate([t])[0] * 100


def _get_eer_in_percent(exp: Experiment) -> float:
    result = exp.load_verification_benchmark_results()
    return result.get_equal_error_rate() * 100


def main():
    # Table for different preprocessing methods
    EXTRACTOR_KEY = "DeepPrint_TexMinu_512"
    DATASET_NAME = "mcyt330_optical"

    ROTATION_MAGNITUDES = [0, 15, 30, 45, 60, 90]
    SHIFT_MAGNITUDES = [10, 20, 30, 40, 60, 80]

    results_mat: np.array = np.zeros((len(ROTATION_MAGNITUDES), len(SHIFT_MAGNITUDES)))
    rw_results_mat: np.array = np.zeros(
        (len(ROTATION_MAGNITUDES), len(SHIFT_MAGNITUDES))
    )
    for i, r in enumerate(ROTATION_MAGNITUDES):
        for j, s in enumerate(SHIFT_MAGNITUDES):
            experiment, rw_experiment = _get_experiment_with_rw(
                DATASET_NAME, EXTRACTOR_KEY, r, s
            )
            results_mat[i, j] = _get_fnmr_at_fmr_in_percent(experiment, 0.1)
            rw_results_mat[i, j] = _get_fnmr_at_fmr_in_percent(rw_experiment, 0.1)

    xlabel = "Max. shift in pixels"
    ylabel = "Max. rotation in degrees"
    ytick_labels = [f"{r}" for r in ROTATION_MAGNITUDES]
    xtick_labels = [f"{s}" for s in SHIFT_MAGNITUDES]

    plotdir = get_figures_dir("alignment", DATASET_NAME)

    for mat, filename in [
        (results_mat, f"{EXTRACTOR_KEY}.png"),
        (rw_results_mat, f"{EXTRACTOR_KEY}_rw.png"),
    ]:
        plot_heatmap(
            mat,
            xtick_labs=xtick_labels,
            ytick_labs=ytick_labels,
            xlabel=xlabel,
            ylabel=ylabel,
            title=None,
            scale_label="FNMR (%) at FMR=0.1%",
            filename=join(plotdir, filename),
        )


if __name__ == "__main__":
    main()
