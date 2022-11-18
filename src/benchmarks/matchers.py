import numpy as np

from src.data.biometric_dataset import Identifier
from src.data.embedding_dataset import EmbeddingDataset
from src.data.biometric_dataset import Identifier


class BiometricMatcher:
    def __init__(self):
        raise NotImplementedError()

    def similarity(self, sample1: Identifier, sample2: Identifier) -> float:
        raise NotImplementedError()


class VectorizedMatcher(BiometricMatcher):
    def preload_vectorized(self, samples: list[Identifier]) -> None:
        """
        Preloads all samples into one numpy ndarray for vectorized comparison.
        """
        raise NotImplementedError()

    def vectorized_similarity(self, sample: Identifier) -> np.ndarray[float]:
        """
        Similarities with all the samples in the preloaded vector.
        """
        raise NotImplementedError()


class CosineSimilarityMatcher(VectorizedMatcher):
    def __init__(self, embedding_dataset: EmbeddingDataset):
        self._embeddings = embedding_dataset
        self._vector = None

    def similarity(self, sample1: Identifier, sample2: Identifier) -> float:
        emb1 = self._embeddings.get(sample1)
        emb2 = self._embeddings.get(sample2)
        return np.dot(emb1.vector, emb2.vector)

    @property
    def embedding_dimensions(self) -> int:
        if len(self._embeddings) == 0:
            return None
        _, _, emb = self._embeddings[0]
        return emb.dimensions

    def preload_vectorized(self, samples: list[Identifier]) -> None:
        """
        Preloads all samples into one numpy ndarray for vectorized comparison.
        """
        self._vector = np.ndarray(shape=(len(samples), self.embedding_dimensions))
        for i, sample in enumerate(samples):
            emb = self._embeddings.get(sample)
            self._vector[i, :] = emb.vector

    def vectorized_similarity(self, sample: Identifier) -> np.ndarray[float]:
        """
        Similarities for all the items in the preloaded vector.
        """
        emb = self._embeddings.get(sample)
        vals = np.matmul(self._vector, emb.vector)
        # Negative similarity makes no sense, as a fingerprint does not have an opposite
        vals[vals < 0] = 0
        return vals