from flx.data.dataset import (
    Identifier,
)


class BiometricComparison:
    def __init__(self, sample1: Identifier, sample2: Identifier):
        self.sample1: Identifier = sample1
        self.sample2: Identifier = sample2

    @property
    def mated(self) -> bool:
        return self.sample1.subject == self.sample2.subject

    def __str__(self) -> str:
        return f"BiometricComparison({self.sample1}, {self.sample2})"


class BiometricComparisonResult:
    def __init__(self, comparison: BiometricComparison, similarity: float):
        self.comparison: BiometricComparison = comparison
        self.similarity: float = similarity

    def __str__(self) -> str:
        return f"BiometricComparisonResult({self.comparison.sample1}, {self.comparison.sample2}, {self.similarity})"


def biometric_comparisons_to_json(comparisons: list[BiometricComparison]) -> None:
    return {
        "array_sample1": Identifier.ids_to_json([comp.sample1 for comp in comparisons]),
        "array_sample2": Identifier.ids_to_json([comp.sample2 for comp in comparisons]),
    }


def biometric_comparisons_from_json(jsn: dict) -> list[BiometricComparison]:
    array_sample1 = Identifier.ids_from_json(jsn["array_sample1"])
    array_sample2 = Identifier.ids_from_json(jsn["array_sample2"])
    return [
        BiometricComparison(sample1, sample2)
        for sample1, sample2 in zip(array_sample1, array_sample2)
    ]


def biometric_comparison_results_to_json(
    path: str, results: list[BiometricComparisonResult]
) -> None:
    return {
        "array_comparison": biometric_comparisons_to_json(
            [res.comparison for res in results]
        ),
        "array_similarity": [float(res.similarity) for res in results],
    }


def biometric_comparison_results_from_json(
    jsn: dict,
) -> list[BiometricComparisonResult]:
    comparisons = biometric_comparisons_from_json(jsn["array_comparison"])
    similarities = jsn["array_similarity"]
    return [
        BiometricComparisonResult(comp, sim)
        for comp, sim in zip(comparisons, similarities)
    ]
