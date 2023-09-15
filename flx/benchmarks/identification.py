import json
import os
from os.path import join

import tqdm
import numpy as np

from flx.data.dataset import Identifier
from flx.benchmarks.matchers import BiometricMatcher, VectorizedMatcher
from flx.benchmarks.biometric_search import (
    ExhaustiveSearch,
    ExhaustiveSearchResult,
    exhaustive_searches_to_json,
    exhaustive_searches_from_json,
    exhaustive_search_results_to_json,
    exhaustive_search_results_from_json,
)


class FoldResult:
    def __init__(self, search_results: list[ExhaustiveSearchResult]):
        assert len(search_results) > 0
        self.gallery: np.ndarray[Identifier] = search_results[0].search.gallery
        self._mated_results: list[ExhaustiveSearchResult] = []
        self._non_mated_results: list[ExhaustiveSearchResult] = []
        for result in search_results:
            assert id(result.search.gallery) == id(self.gallery)
            if result.search.is_mated:
                self._mated_results.append(result)
            else:
                self._non_mated_results.append(result)
        self._mated_ranks: list[int] = [s.rank for s in self._mated_results]
        self._mated_similarities: np.ndarray = [
            s.similarity for s in self._mated_results
        ]
        self._non_mated_similarities: np.ndarray = [
            s.similarity for s in self._non_mated_results
        ]

    def false_positive_identification_rate(self, threshold: float) -> float:
        """
        Returns the rate of non-mated searches where an enrolled reference is returned.
        """
        assert len(self._non_mated_results) > 0
        return np.sum(self._non_mated_similarities >= threshold) / len(
            self._non_mated_results
        )

    def false_negative_identification_rate(
        self, threshold: float = None, fpir: float = None
    ) -> float:
        """
        Returns the rate of mated searches where the probe was not in the candidate list given the threshold.

        Can be used with either a fixed threshold or a number of candidates.
        """
        if threshold is not None:
            return np.sum(self._mated_similarities < threshold) / len(
                self._mated_results
            )
        assert fpir is not None
        nms_sorted = np.sort(self._non_mated_similarities)
        num_fm = int(len(self._non_mated_similarities) * fpir) + 1
        return self.false_negative_identification_rate(threshold=nms_sorted[-num_fm])

    def get_mated_ranks(self) -> list[int]:
        return self._mated_ranks

    def get_mated_similarities(self) -> list[float]:
        return self._mated_similarities

    def get_highest_non_mated_similarities(self) -> list[float]:
        return self._non_mated_similarities

    def save(self, path: str) -> None:
        results = self._mated_results + self._non_mated_results
        jsn = exhaustive_search_results_to_json(results)
        with open(os.path.join(path, "searches.json"), "w") as f:
            json.dump(jsn, f)

    @staticmethod
    def load(path: str) -> dict:
        with open(os.path.join(path, "searches.json"), "r") as f:
            jsn = json.load(f)
        results = exhaustive_search_results_from_json(jsn)
        return FoldResult(results)


class IdentificationResult:
    def __init__(self, results: list[FoldResult]):
        self._results = results

    def false_positive_identification_rate(self, threshold: float) -> float:
        rates = [
            result.false_positive_identification_rate(threshold)
            for result in self._results
        ]
        return sum(rates) / len(rates)

    def false_negative_identification_rate(
        self, threshold: float = None, fpir: int = None
    ) -> float:
        rates = [
            result.false_negative_identification_rate(threshold=threshold, fpir=fpir)
            for result in self._results
        ]
        return sum(rates) / len(rates)

    def get_mated_ranks(self) -> list[int]:
        # return the concatenated list of mated ranks for each result
        return [rank for result in self._results for rank in result.get_mated_ranks()]

    def get_mated_similarities(self) -> list[float]:
        # return the concatenated list of mated similarities for each result
        return [
            sim for result in self._results for sim in result.get_mated_similarities()
        ]

    def get_highest_non_mated_similarities(self) -> list[float]:
        # return the concatenated list of highest non-mated similarities for each result
        return [
            sim
            for result in self._results
            for sim in result.get_highest_non_mated_similarities()
        ]

    def save(self, path: str) -> None:
        for i, r in enumerate(self._results):
            fold_dir = os.path.join(path, f"fold_{i}")
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)
            r.save(fold_dir)

    @staticmethod
    def load(path: str) -> dict:
        n_folds = len([f.name for f in os.scandir(path) if f.is_dir()])
        if n_folds == 0:
            raise RuntimeError(
                f"Cannot load identification results from {path}: The directory is empty!"
            )
        print(f"Loading identification results for {n_folds} cross-validation folds")
        return IdentificationResult(
            [FoldResult.load(join(path, f"fold_{i}")) for i in range(n_folds)]
        )


class IdentificationBenchmark:
    def __init__(self, folds: list[list[ExhaustiveSearch]]):
        self._folds: list[list[ExhaustiveSearch]] = folds

    def _run_single_fold(self, matcher: BiometricMatcher, fold: int) -> FoldResult:
        assert 0 <= fold and fold < len(self._folds)
        searches = self._folds[fold]

        gallery: np.ndarray[Identifier] = searches[0].gallery
        for s in searches:
            assert id(s.gallery) == id(gallery)

        results = []
        if isinstance(matcher, VectorizedMatcher):
            print("Running benchmark vectorized")
            matcher.preload_vectorized(gallery)
            for search in tqdm.tqdm(searches):
                similarities = matcher.vectorized_similarity(search.probe)
                results.append(
                    ExhaustiveSearchResult.from_similarity_scores(search, similarities)
                )
        else:
            print("Running non-vectorized")
            for search in tqdm.tqdm(searches):
                similarities = np.array(
                    [matcher.similarity(search.probe, sample) for sample in gallery]
                )
                results.append(
                    ExhaustiveSearchResult.from_similarity_scores(search, similarities)
                )
        return FoldResult(results)

    def run(self, matcher: BiometricMatcher) -> IdentificationResult:
        return IdentificationResult(
            [self._run_single_fold(matcher, i) for i in range(len(self._folds))]
        )

    def save(self, path: str) -> None:
        jsn = [exhaustive_searches_to_json(s) for s in self._folds]
        with open(path, "w") as f:
            json.dump(jsn, f)

    @staticmethod
    def load(path: str) -> "IdentificationBenchmark":
        with open(path, "r") as f:
            jsn = json.load(f)
        folds = [exhaustive_searches_from_json(s) for s in jsn]
        return IdentificationBenchmark(folds=folds)
