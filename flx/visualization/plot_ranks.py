import tqdm
import numpy as np
from matplotlib import pyplot as plt
import seaborn

from flx.benchmarks.identification import IdentificationResult

MAX_N = 20


def _get_rank_n_identification_rates(
    mated_ranks: list[int], max_n: int
) -> tuple[list[float], list[float]]:
    ranks_array = np.array(mated_ranks, dtype=int)
    identification_rates = []
    for n in tqdm.tqdm(range(1, max_n + 1)):
        rank_n_or_lower = np.sum(ranks_array <= n)
        identification_rates.append(rank_n_or_lower / len(mated_ranks))
    return identification_rates


def plot_rank_n_identification_rates(
    path: str,
    results: list[IdentificationResult],
    model_labels: list[str],
    plot_title: str,
) -> None:
    plt.close()  # In case some other function was not tidy
    seaborn.set_style("whitegrid")
    ax = plt.subplot()
    for label, res in zip(model_labels, results):
        identification_rates = _get_rank_n_identification_rates(
            res.get_mated_ranks(), MAX_N
        )
        seaborn.lineplot(
            x=list(range(1, MAX_N + 1)),
            y=[r * 100 for r in identification_rates],
            label=label,
            ax=ax,
        )

    plt.title(plot_title)
    plt.legend(loc="lower right")
    plt.xscale("linear")
    plt.xticks(list(range(1, MAX_N + 1)), [str(i) for i in range(1, MAX_N + 1)])
    plt.yscale("linear")
    plt.xlabel("Rank", fontweight="bold")
    plt.ylabel("Identificiation rate (in %)", fontweight="bold")
    plt.savefig(path + ".png")
    plt.savefig(path + ".pdf")
    plt.close()
