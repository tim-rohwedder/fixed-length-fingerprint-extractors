import numpy as np

from flx.data.dataset import (
    Identifier,
)


class ExhaustiveSearch:
    """
    Uses numpy arrays for efficiency. The gallery should never be modified as all Searches share a reference to
    the same gallery.

    @param probe : identifier of the probe
    @param gallery : identifiers of biometric references in the enrolment dataset
    @param similarities : similarity scores for comparisons with the references in gallery
        can be left None when the search is created, but must be set before accessing the search statistics.

    """

    def __init__(
        self, probe: Identifier, gallery: np.ndarray[Identifier], is_mated: bool
    ):
        self.probe: Identifier = probe
        self.is_mated: bool = is_mated
        self.gallery: np.ndarray[Identifier] = gallery


def _calculate_rank(search: ExhaustiveSearch, similarities: np.ndarray):
    if not search.is_mated:
        return -1
    idxs_sorted: np.ndarray = np.argsort(similarities)
    rank = 0
    for idx in np.flip(idxs_sorted)[:]:
        rank += 1
        if search.gallery[idx].subject == search.probe.subject:
            return rank


class ExhaustiveSearchResult:
    def __init__(self, search: ExhaustiveSearch, similarity: float, rank: int):
        """
        'rank' is the position of the first mated comparison in a sorted list of all comparisons in the query
        (sorted by descending similarity).
        If the search was non-mated -1 is returned.
        """
        self.search: ExhaustiveSearch = search
        self.rank: int = rank
        self.similarity: float = similarity

    def is_positive_identification(self, threshold: float) -> bool:
        """
        Returns whether the identification decision using the specified threshold is positive or negative.
        """
        return threshold <= self.similarity

    @staticmethod
    def from_similarity_scores(
        search: ExhaustiveSearch, gallery_similarities: np.ndarray[np.float16]
    ) -> "ExhaustiveSearchResult":
        return ExhaustiveSearchResult(
            search,
            rank=_calculate_rank(search, gallery_similarities),
            similarity=float(np.amax(gallery_similarities)),
        )


def exhaustive_searches_to_json(searches: list[ExhaustiveSearch]) -> dict:
    assert len(searches) > 0
    gallery: np.ndarray[Identifier] = searches[0].gallery
    probes: list[Identifier] = [s.probe for s in searches]
    return {
        "gallery": Identifier.ids_to_json(gallery),
        "probes": Identifier.ids_to_json(probes),
        "mated": [s.is_mated for s in searches],
    }


def exhaustive_searches_from_json(jsn: dict) -> list[ExhaustiveSearch]:
    gallery: np.ndarray[Identifier] = np.squeeze(
        np.array(Identifier.ids_from_json(jsn["gallery"]))
    )
    probes: list[Identifier] = Identifier.ids_from_json(jsn["probes"])
    mated: list[bool] = jsn["mated"]
    return [
        ExhaustiveSearch(probe=p, gallery=gallery, is_mated=m)
        for p, m in zip(probes, mated)
    ]


def exhaustive_search_results_to_json(results: list[ExhaustiveSearchResult]) -> dict:
    return {
        "searches": exhaustive_searches_to_json([r.search for r in results]),
        "ranks": [r.rank for r in results],
        "similarities": [r.similarity for r in results],
    }


def exhaustive_search_results_from_json(jsn: dict) -> list[ExhaustiveSearchResult]:
    searches: list[ExhaustiveSearch] = exhaustive_searches_from_json(jsn["searches"])
    ranks: list[int] = jsn["ranks"]
    similarities: list[float] = jsn["similarities"]
    assert len(searches) == len(ranks)
    assert len(searches) == len(similarities)
    return [
        ExhaustiveSearchResult(se, rank=r, similarity=si)
        for se, r, si in zip(searches, ranks, similarities)
    ]
