from typing import Union
from dataclasses import dataclass

import os

from flx.setup.paths import (
    get_verification_benchmark_file,
    get_verification_benchmark_results_file,
    get_closed_set_benchmark_file,
    get_open_set_benchmark_file,
    get_closed_set_benchmark_results_dir,
    get_open_set_benchmark_results_dir,
    get_generated_embeddings_dir,
    get_texture_embedding_dataset_dir,
    get_minutia_embedding_dataset_dir,
    get_reweighted_embedding_dataset_dir,
)

from flx.benchmarks.matchers import CosineSimilarityMatcher
from flx.benchmarks.verification import VerificationBenchmark, VerificationResult
from flx.benchmarks.identification import IdentificationBenchmark, IdentificationResult
from flx.data.embedding_loader import EmbeddingLoader


@dataclass
class Experiment:
    model_name: str
    model_label: str
    dataset_name: str
    dataset_label: str
    reweighting_training_indices: list[int] = None

    @staticmethod
    def _load_embeddings_if_exist(path: str) -> Union[None, EmbeddingLoader]:
        if not os.path.exists(path) or len(os.listdir(path)) == 0:
            return None
        return EmbeddingLoader.load(path)

    def load_training_embeddings(self) -> EmbeddingLoader:
        assert self.reweighting_training_indices is not None
        all_embeddings = self.load_embeddings()
        train_ids = (
            all_embeddings.ids[idx] for idx in self.reweighting_training_indices
        )
        return EmbeddingLoader([all_embeddings.get(bid) for bid in train_ids])

    def load_embeddings(self) -> EmbeddingLoader:
        embeddings_base_dir = get_generated_embeddings_dir(
            self.model_name, self.dataset_name
        )
        tex_embeddings = self._load_embeddings_if_exist(
            get_texture_embedding_dataset_dir(embeddings_base_dir)
        )
        minu_embeddings = self._load_embeddings_if_exist(
            get_minutia_embedding_dataset_dir(embeddings_base_dir)
        )
        return EmbeddingLoader.combine_if_both_exist(tex_embeddings, minu_embeddings)

    def save_embeddings(
        self,
        texture_embeddings: EmbeddingLoader,
        minutia_embeddings: EmbeddingLoader,
    ):
        embeddings_base_dir = get_generated_embeddings_dir(
            self.model_name, self.dataset_name
        )
        if texture_embeddings is not None:
            texture_embeddings.save(
                get_texture_embedding_dataset_dir(embeddings_base_dir)
            )
        if minutia_embeddings is not None:
            minutia_embeddings.save(
                get_minutia_embedding_dataset_dir(embeddings_base_dir)
            )

    def load_verification_benchmark(self) -> VerificationBenchmark:
        return VerificationBenchmark.load(
            get_verification_benchmark_file(self.dataset_name)
        )

    def load_verification_benchmark_results(self) -> VerificationResult:
        try:
            return VerificationResult.load(
                get_verification_benchmark_results_file(
                    self.model_name, self.dataset_name
                )
            )
        except RuntimeError:
            benchmark = self.load_verification_benchmark()
            matcher = CosineSimilarityMatcher(self.load_embeddings())
            results = benchmark.run(matcher, save=True)
            self.save_verification_benchmark_results(results)
            return results

    def save_verification_benchmark_results(self, results: VerificationResult) -> None:
        return results.save(
            get_verification_benchmark_results_file(self.model_name, self.dataset_name)
        )

    def load_closed_set_benchmark(self) -> IdentificationBenchmark:
        return IdentificationBenchmark.load(
            get_closed_set_benchmark_file(self.dataset_name)
        )

    def load_closed_set_benchmark_results(self) -> IdentificationResult:
        try:
            return IdentificationResult.load(
                get_closed_set_benchmark_results_dir(self.model_name, self.dataset_name)
            )
        except RuntimeError:
            benchmark = self.load_closed_set_benchmark()
            matcher = CosineSimilarityMatcher(self.load_embeddings())
            results = benchmark.run(matcher)
            self.save_closed_set_benchmark_results(results)
            return results

    def save_closed_set_benchmark_results(self, results: IdentificationResult) -> None:
        return results.save(
            get_closed_set_benchmark_results_dir(self.model_name, self.dataset_name)
        )

    def load_open_set_benchmark(self) -> IdentificationBenchmark:
        return IdentificationBenchmark.load(
            get_open_set_benchmark_file(self.dataset_name)
        )

    def load_open_set_benchmark_results(self) -> IdentificationResult:
        try:
            return IdentificationResult.load(
                get_open_set_benchmark_results_dir(self.model_name, self.dataset_name)
            )
        except RuntimeError:
            benchmark = self.load_open_set_benchmark()
            matcher = CosineSimilarityMatcher(self.load_embeddings())
            results = benchmark.run(matcher)
            self.save_open_set_benchmark_results(results)
            return results

    def save_open_set_benchmark_results(self, results: IdentificationResult) -> None:
        return results.save(
            get_open_set_benchmark_results_dir(self.model_name, self.dataset_name)
        )


class ReweightingExperiment(Experiment):
    REWEIGHTED_EXTRACTOR_NAME_POSTFIX = "_reweighted"
    REWEIGHTED_EXTRACTOR_LABEL_POSTFIX = " (reweighted)"

    def __init__(self, model_name: str, model_label: str, **kwargs):
        super().__init__(
            model_name=model_name + self.REWEIGHTED_EXTRACTOR_NAME_POSTFIX,
            model_label=model_label + self.REWEIGHTED_EXTRACTOR_LABEL_POSTFIX,
            **kwargs
        )

    def load_embeddings(self) -> EmbeddingLoader:
        embeddings_base_dir = get_generated_embeddings_dir(
            self.model_name[: -len(self.REWEIGHTED_EXTRACTOR_NAME_POSTFIX)],
            self.dataset_name,
        )
        return EmbeddingLoader.load(
            get_reweighted_embedding_dataset_dir(embeddings_base_dir)
        )

    def save_embeddings(
        self,
        reweighted_embeddings: EmbeddingLoader,
    ):
        embeddings_base_dir = get_generated_embeddings_dir(
            self.model_name[: -len(self.REWEIGHTED_EXTRACTOR_NAME_POSTFIX)],
            self.dataset_name,
        )
        reweighted_embeddings.save(
            get_reweighted_embedding_dataset_dir(embeddings_base_dir)
        )
