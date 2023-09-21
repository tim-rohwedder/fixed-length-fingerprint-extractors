from typing import Union
from os.path import join
import random
import tqdm
import numpy as np

from flx.setup.paths import (
    get_verification_benchmark_file,
    get_closed_set_benchmark_file,
    get_open_set_benchmark_file,
    BENCHMARKS_DIR,
)


from flx.data.dataset import Identifier
from flx.data.image_loader import MCYTCapacitiveLoader, MCYTOpticalLoader
from flx.benchmarks.biometric_comparison import BiometricComparison
from flx.benchmarks.biometric_search import ExhaustiveSearch
from flx.benchmarks.verification import VerificationBenchmark
from flx.benchmarks.identification import IdentificationBenchmark
from flx.setup.paths import get_fingerprint_dataset_path


def create_verification_benchmark(subjects: list[int], impressions_per_subject: list[int]) -> VerificationBenchmark:
    """
    Compare each mated pair and for each sample do as many non-mated comparisons as there are impressions per sample.
    """
    random.seed(3984)
    comps_mated = []
    for i in subjects:
        for idx, j in enumerate(impressions_per_subject):
            comps_mated += [
                BiometricComparison(Identifier(i, j), Identifier(i, k))
                for k in impressions_per_subject[idx + 1 :]
            ]

    comps_non_mated = []
    the_grid: list[set] = {
        x: set((i, k) for k in impressions_per_subject for i in subjects if i != x)
        for x in subjects
    }
    for impression2 in impressions_per_subject:
        for subject1 in tqdm.tqdm(subjects):
            baseset = the_grid[subject1]
            js = random.sample(list(baseset), len(impressions_per_subject))
            for subject2, impression2 in js:
                comps_non_mated.append(
                    BiometricComparison(
                        Identifier(subject1, impression2),
                        Identifier(subject2, impression2),
                    )
                )
                try:
                    the_grid[subject2].remove((subject1, impression2))
                except:
                    pass
    return VerificationBenchmark(comps_mated + comps_non_mated)


def _make_verification(
    name: str, subjects: list[int], impressions_per_subject: list[int]
):
    print(f"Making verification benchmark: {name}")
    bm = create_verification_benchmark(subjects, impressions_per_subject)
    bm.save(get_verification_benchmark_file(name))
    

def create_identification_benchmark(
    subjects_gallery: list[int],
    subjects_impostor: list[int],
    impressions: list[int],
    gallery_impression_idx: int = 0,
) -> list[ExhaustiveSearch]:
    """
    Makes a gallery containing the first impression [first element of 'impressions', of each of the 'subjects_gallery'.
    For each of the subjects in the gallery and for each except the first impressions one query is added
    For each of the impostor subjects one query is added per impression in range [0, impressions_per_subject - 1]
    """
    random.seed(3984)
    assert len(impressions) > 1
    assert len(subjects_gallery) > 0
    searches = []
    gallery_samples = np.array(
        [Identifier(s, impressions[gallery_impression_idx]) for s in subjects_gallery]
    )
    for s in subjects_gallery:
        searches += [
            ExhaustiveSearch(Identifier(s, i), gallery_samples, True)
            for i in impressions[1:]
        ]
    for s in subjects_impostor:
        searches += [
            ExhaustiveSearch(Identifier(s, i), gallery_samples, False)
            for i in impressions
        ]
    return searches


def _make_identification_closed_set(
    name: str,
    subjects_gallery: list[int],
    impressions: list[int],
) -> IdentificationBenchmark:
    print(f"make closed-set identification: {name}")
    searches = [create_identification_benchmark(
        subjects_gallery, [], impressions, gallery_impression_idx=i
    ) for i in range(len(impressions))]
    bm = IdentificationBenchmark(searches)
    bm.save(get_closed_set_benchmark_file(name))


def _make_identification_open_set(
    name: str, subjects: list[int], impressions: list[int], folds: int = 10
):
    random.shuffle(subjects)

    l = len(subjects) // 10
    sublists = [subjects[i * l : min((i + 1) * l, len(subjects))] for i in range(folds)]

    folds: list[IdentificationBenchmark] = []
    for impostor in tqdm.tqdm(sublists):
        gallery = [
            item for sublist in sublists if sublist != impostor for item in sublist
        ]
        folds.append(create_identification_benchmark(gallery, impostor, impressions))
    bm = IdentificationBenchmark(folds)
    bm.save(get_open_set_benchmark_file(name))


def _make_identification(*args, **kwargs) -> IdentificationBenchmark:
    _make_identification_closed_set(*args, **kwargs)
    _make_identification_open_set(*args, **kwargs)


def make_benchmarks_SFinge():
    # Verification
    _make_verification(
        "SFingev2ValidationSeparateSubjects", list(range(42000, 44000)), list(range(4))
    )
    # Identification
    _make_identification(
        "SFingev2ValidationSeparateSubjects",
        list(range(42000, 44000)),
        list(range(4)),
    )


def _make_verification_FVC2004():
    """
    Expects the file with the FVC2004 comparisons (mated and non-mated) as input.
    """
    print("make_verification_FVC2004")
    comps = []
    with open(join(BENCHMARKS_DIR, "verification", "comparisons_FVC2004.txt")) as file:
        l = 1
        for line in file.readlines():
            l += 1
            a, b = line.split(" ")
            a_sid = int(a.split("_")[0]) - 1
            a_iid = int(a.split("_")[1].split(".")[0]) - 1
            aid = Identifier(subject=a_sid, impression=a_iid)
            b_sid = int(b.split("_")[0]) - 1
            b_iid = int(b.split("_")[1].split(".")[0]) - 1
            bid = Identifier(subject=b_sid, impression=b_iid)
            comps.append(BiometricComparison(aid, bid))

    bm = VerificationBenchmark(comps)
    bm.save(get_verification_benchmark_file("FVC2004_DB1A"))


def make_benchmarks_FVC2004():
    """
    Verification:
        Uses the comparisons published with FVC2004
    Ident. closed set:
        One sample of each subject in gallery. 7 other samples as probes
    Ident. open set:
        One sample from first 50 subjects as gallery and rest as impostor
    """
    _make_verification_FVC2004
    _make_identification("FVC2004_DB1A", list(range(100)), list(range(8)))


def make_benchmarks_NISTSD4():
    """
    Verification:
        Compare each mated pair and for each sample do ten non-mated comparisons randomly.
    Ident. closed set:
       Use first sample of each subject in gallery and second as probe
    Ident. open set:
       Use first sample of first 1000 subjects in gallery and rest as probes
    """
    _make_verification("NIST SD4", range(2000), range(2))
    _make_identification("NIST SD4", list(range(2000)), list(range(2)))


def make_benchmarks_mcyt():
    """
    Uses last 1300 subjects (of total 3300)

    Verification:
        Compare each mated pair and for each sample do ten non-mated comparisons randomly.
    Ident. closed set:
       Use first sample of each subject in gallery and second as probe
    Ident. open set:
       Use first sample of last 250 subjects in gallery and rest as probes
    """
    N_TOTAL = 3300
    N_LAST = 1300
    # Verification
    def get_subjects(bids):
        return sorted(list({id.subject for id in bids}))

    ds_optical = MCYTOpticalLoader(get_fingerprint_dataset_path("mcyt330_optical"))
    ds_capacitive = MCYTCapacitiveLoader(
        get_fingerprint_dataset_path("mcyt330_capacitive")
    )
    optical_subjects = get_subjects(ds_optical.ids)
    capacitive_subjects = get_subjects(ds_capacitive.ids)
    optical_subjects = optical_subjects[N_TOTAL - N_LAST :]
    capacitive_subjects = capacitive_subjects[N_TOTAL - N_LAST :]
    assert len(optical_subjects) == N_LAST
    assert len(capacitive_subjects) == N_LAST
    _make_verification("mcyt330_optical", optical_subjects, list(range(12)))
    _make_verification("mcyt330_capacitive", capacitive_subjects, list(range(12)))
    _make_identification(
        "mcyt330_optical",
        optical_subjects,
        list(range(12)),
    )
    _make_identification(
        "mcyt330_capacitive",
        capacitive_subjects,
        list(range(12)),
    )


def main():
    make_benchmarks_SFinge()
    make_benchmarks_mcyt()


if __name__ == "__main__":
    main()
