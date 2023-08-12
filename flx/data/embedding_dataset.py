import json
from os.path import join
from copy import copy

import numpy as np

from flx.data.biometric_dataset import Identifier, IdentifierSet, BiometricDataset


class BiometricEmbedding:
    def __init__(self, identifier: Identifier, embedding: np.ndarray):
        assert len(embedding.shape) == 1
        self.id = identifier
        self.vector = embedding / np.linalg.norm(embedding, ord=2)

    @property
    def dimensions(self) -> int:
        return self.vector.shape[0]

    def concat(self, other: "BiometricEmbedding") -> "BiometricEmbedding":
        if self.id != other.id:
            raise ValueError(
                "Can only concatenate embeddings that belong to the same sample!"
            )
        return BiometricEmbedding(
            copy(self.id), np.concatenate([self.vector, other.vector])
        )


class EmbeddingDataset(BiometricDataset):
    ELEMENT_TYPE = BiometricEmbedding

    def __init__(self, embeddings: list[BiometricEmbedding]):
        self.embeddings: dict[Identifier, BiometricEmbedding] = {
            e.id: e for e in embeddings
        }
        self._ids = IdentifierSet(list(self.embeddings.keys()))

    @property
    def ids(self) -> list[Identifier]:
        return self._ids

    def get(self, bid: Identifier) -> BiometricEmbedding:
        return self.embeddings[bid]

    def save(self, outdir: str) -> None:
        """
        Saves embeddings and corresponding ids to disk

        @param outdir : absolute path of the output directory
        """
        outarr = np.vstack(
            [self.embeddings[identifier].vector for identifier in self._ids]
        )
        outarr = outarr.astype(np.float32)
        np.save(join(outdir, "embeddings.npy"), outarr)
        with open(join(outdir, "ids.json"), "w") as file:
            json.dump(Identifier.ids_to_json(self._ids), file, indent=None)

    @staticmethod
    def load(dir: str) -> "EmbeddingDataset":
        """
        Loads embedding dataset from disk

        @dir : absolute path of the embeddings dir

        @returns : Loaded embeddings dataset
        """
        with open(join(dir, "ids.json"), "r") as file:
            ids = Identifier.ids_from_json(json.load(file))
        inarr = np.load(join(dir, "embeddings.npy"))
        return EmbeddingDataset(
            [BiometricEmbedding(bid, emb) for bid, emb in zip(ids, inarr)]
        )

    def numpy(self) -> np.ndarray:
        return np.vstack([e.vector for e in self.embeddings.values()])


def combine_embeddings(
    ds1: EmbeddingDataset, ds2: EmbeddingDataset
) -> EmbeddingDataset:
    if set(ds1.ids) != set(ds2.ids):
        raise ValueError(
            "To concatenate two EmbeddingDatasets they must contain the same ids!"
        )
    concatenated = [
        BiometricEmbedding(i, np.concatenate([ds1.get(i).vector, ds2.get(i).vector]))
        for i in ds1.ids
    ]
    return EmbeddingDataset(concatenated)


def combine_embeddings_if_both_exist(
    ds1: EmbeddingDataset, ds2: EmbeddingDataset
) -> EmbeddingDataset:
    if ds1 is None:
        assert ds2 is not None
        return ds2
    if ds2 is None:
        assert ds1 is not None
        return ds1
    return combine_embeddings(ds1, ds2)