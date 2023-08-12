import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from flx.benchmarks.verification import VerificationResult


def _plot_kde(
    distributions: list[np.ndarray], labels: list[str], title: str, filename: str
) -> None:
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot()
    for dis, l in zip(distributions, labels):
        sns.histplot(dis, ax=ax, label=f"{l} (n = {dis.shape[0]})")
    if title is not None:
        plt.title(title)
    fig.tight_layout()
    plt.legend()
    plt.savefig(filename)
    plt.close()


def plot_similarity_scores_results(
    paths: str, results: list[VerificationResult], plot_titles: list[str]
):
    assert len(paths) == len(results)
    assert len(paths) == len(plot_titles)
    for path, res, title in zip(paths, results, plot_titles):
        _plot_kde(
            distributions=[res.get_mated_scores(), res.get_non_mated_scores()],
            labels=["Genuine", "Impostor"],
            title=title,
            filename=path,
        )
