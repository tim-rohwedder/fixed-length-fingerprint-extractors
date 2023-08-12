import numpy as np
from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import normalize, StandardScaler

import numpy as np
from itertools import groupby


def _mated_pairs(indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    sorted_indices = indices[np.argsort(labels)]
    sorted_labels = labels[np.argsort(labels)]
    pairs = []
    for label, group in groupby(zip(sorted_indices, sorted_labels), key=lambda x: x[1]):
        group_indices = [x[0] for x in group]
        for i in range(len(group_indices)):
            for j in range(i + 1, len(group_indices)):
                pairs.append((group_indices[i], group_indices[j]))
    return np.array(pairs)


def _non_mated_pairs(indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    np.random.seed(50)
    unique_labels = np.unique(labels)
    pairs = []
    for label in unique_labels:
        label_indices = indices[labels == label]
        n = len(label_indices)
        other_indices = indices[labels != label]
        for _ in range(n * (n - 1)):
            i = np.random.choice(label_indices)
            j = np.random.choice(other_indices)
            pairs.append((i, j))
    return np.array(pairs)


def _pairwise_elementwise_product(
    embeddings: np.ndarray, index_pairs: np.ndarray
) -> np.ndarray:
    i_indices = index_pairs[:, 0]
    j_indices = index_pairs[:, 1]
    i_embeddings = embeddings[i_indices]
    j_embeddings = embeddings[j_indices]
    return i_embeddings * j_embeddings


def _pairwise_target_similarity(
    labels: np.ndarray, index_pairs: np.ndarray
) -> np.ndarray:
    i_indices = index_pairs[:, 0]
    j_indices = index_pairs[:, 1]
    i_labels = labels[i_indices]
    j_labels = labels[j_indices]
    return i_labels == j_labels


def _reweight_embedding_dimensions(embeddings: np.ndarray, w: np.ndarray) -> np.ndarray:
    assert w.ndim == 1, "w must be a 1D array"
    assert (
        embeddings.shape[1] == w.shape[0]
    ), "Number of columns in embeddings must match the length of w"

    w = np.maximum(0, w)
    return embeddings * np.sqrt(w)


def _linear_regression(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    indices = np.array(range(embeddings.shape[0]))
    index_pairs = np.concatenate(
        [_mated_pairs(indices, labels), _non_mated_pairs(indices, labels)], axis=0
    )
    X = _pairwise_elementwise_product(embeddings, index_pairs)
    y = _pairwise_target_similarity(labels, index_pairs)
    clf = SGDClassifier()
    clf.fit(X, y)
    return np.squeeze(clf.coef_)


def reweight_and_normalize_embeddings(
    training_embeddings: np.ndarray,
    embeddings_to_reweight: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # Create a StandardScaler object and fit the scaler on the first array
    scaler = StandardScaler()
    scaler.fit(training_embeddings)
    training_embeddings: np.ndarray = scaler.transform(training_embeddings)
    embeddings_to_reweight: np.ndarray = scaler.transform(embeddings_to_reweight)

    w: np.ndarray = _linear_regression(training_embeddings, labels)
    reweighted_embeddings: np.ndarray = _reweight_embedding_dimensions(
        embeddings_to_reweight, w
    )
    normalized_embeddings: np.ndarray = normalize(reweighted_embeddings, norm="l2")
    return normalized_embeddings
