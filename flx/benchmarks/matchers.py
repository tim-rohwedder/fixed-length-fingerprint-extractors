from abc import ABC, abstractmethod

import numpy as np

from flx.data.dataset import Identifier
from flx.data.embedding_loader import EmbeddingLoader
from flx.data.dataset import Identifier


class BiometricMatcher(ABC):
    @abstractmethod
    def similarity(self, sample1: Identifier, sample2: Identifier) -> float:
        raise NotImplementedError()


class VectorizedMatcher(BiometricMatcher):
    @abstractmethod
    def preload_vectorized(self, samples: list[Identifier]) -> None:
        """
        Preloads all samples into one numpy ndarray for vectorized comparison.
        """
        raise NotImplementedError()

    @abstractmethod
    def vectorized_similarity(self, sample: Identifier) -> np.ndarray[float]:
        """
        Similarities with all the samples in the preloaded vector.
        """
        raise NotImplementedError()


class CosineSimilarityMatcher(VectorizedMatcher):
    def __init__(self, embedding_dataset: EmbeddingLoader):
        self._embeddings = embedding_dataset
        self._matrix = None

    def similarity(self, sample1: Identifier, sample2: Identifier) -> float:
        emb1 = self._embeddings.get(sample1)
        emb2 = self._embeddings.get(sample2)
        return np.dot(emb1, emb2)

    def preload_vectorized(self, samples: list[Identifier]) -> None:
        """
        Preloads all samples into one numpy ndarray for vectorized comparison.
        """
        vectors = [self._embeddings.get(s) for s in samples]
        self._matrix = np.stack(vectors)

    def vectorized_similarity(self, sample: Identifier) -> np.ndarray[float]:
        """
        Similarities for all the items in the preloaded vector.
        """
        emb = self._embeddings.get(sample)
        vals = np.matmul(self._matrix, emb.vector)
        # Negative similarity makes no sense, as a fingerprint does not have an opposite
        vals[vals < 0] = 0
        return vals
