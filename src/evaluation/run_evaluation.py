from paths import (
    get_verification_benchmark_results_file,
    get_identification_benchmark_results_file,
)
from src.benchmarks.matchers import CosineSimilarityMatcher
from src.benchmarks.verification import VerificationBenchmark
from src.benchmarks.identification import IdentificationBenchmark
from src.data.embedding_dataset import EmbeddingDataset


def run_evaluation(trained_model_name: str, dataset_name: str) -> None:
    embeddings = EmbeddingDataset(
        trained_model_name=trained_model_name, eval_dataset_name=dataset_name
    )
    matcher = CosineSimilarityMatcher(embeddings)

    verification = VerificationBenchmark.load(dataset_name)
    results = verification.run(matcher)
    results.save(
        get_verification_benchmark_results_file(
            trained_model_name=trained_model_name, benchmark_name=dataset_name
        )
    )

    identification = IdentificationBenchmark.load(dataset_name)
    results = identification.run(matcher)
    results.save(
        get_identification_benchmark_results_file(
            trained_model_name=trained_model_name, benchmark_name=dataset_name
        )
    )
