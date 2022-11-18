from typing import Hashable, Iterable
import abc
from copy import deepcopy
from collections import defaultdict


class Identifier:
    """
    An identifier for data related to a fingerprint sample.
        subject: An identifier for a specific fingerprint
        impression: An identifier for a sample taken of a fingerprint. E.g. if there are three different samples
            of the same fingerprint they all have the same value for 'subject' but different values for 'impression'
        finger: Optional metadata that tells us which of the ten fingers the subject is. This is needed to determine
            the filename corresponding to a specific sample in some datasets.
    """

    def __init__(self, subject: int, impression: int, finger: int = -1):
        self.subject: int = int(subject)
        self.finger: int = int(finger)
        self.impression: int = int(impression)

    def __hash__(self):
        return hash((self.subject, self.impression))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self) -> str:
        return f"Identifier({self.subject}, {self.impression})"

    def __lt__(self, other: "Identifier"):
        return self.subject < other.subject or (
            self.subject == other.subject and self.impression < other.impression
        )

    @staticmethod
    def ids_to_json(ids: Iterable["Identifier"]) -> dict:
        return {
            "array_subject": [biom_id.subject for biom_id in ids],
            "array_impression": [biom_id.impression for biom_id in ids],
        }

    @staticmethod
    def ids_from_json(jsn: dict) -> Iterable["Identifier"]:
        subjects = jsn["array_subject"]
        samples = jsn["array_impression"]
        assert len(subjects) == len(samples)
        return [Identifier(subject=s, impression=i) for s, i in zip(subjects, samples)]


class IdentifierSet:
    def __init__(self, all_ids: list[Identifier]):
        IdentifierSet._check_duplicates(all_ids)
        self._ids: list[Identifier] = sorted(all_ids)
        self._num_subjects: int = len(set(i.subject for i in self._ids))

        print(
            f"Created IdentifierSet with {self._num_subjects} "
            f"subjects and a total of {len(self)} samples."
        )

    def __getitem__(self, index: int) -> Identifier:
        return self._ids[index]

    def __len__(self) -> int:
        return len(self._ids)

    def __le__(self, other: "IdentifierSet") -> bool:
        return set(self.ids) <= set(other.ids)

    def __eq__(self, other: "IdentifierSet") -> bool:
        return set(self.ids) == set(other.ids)

    @property
    def ids(self) -> list[Identifier]:
        return self._ids

    @property
    def num_subjects(self) -> int:
        return self._num_subjects

    def filter_by_index(self, indices: list[int]) -> "IdentifierSet":
        IdentifierSet._check_duplicates(indices)
        if not set(indices) <= set(i for i in range(len(self))):
            raise ValueError(
                "The indices must be a subset of the indices in the dataset!"
            )
        return IdentifierSet([self._ids[i] for i in indices])

    def filter_by_id(self, ids: "IdentifierSet") -> "IdentifierSet":
        """
        Checks that the ids are a subset of self
        """
        IdentifierSet._check_duplicates(ids)
        if not ids <= self:
            raise ValueError("The ids must be a subset of the ids in the dataset!")
        return ids

    def filter_by_subject(self, subjects: list[int]) -> "IdentifierSet":
        IdentifierSet._check_duplicates(subjects)
        return IdentifierSet._filter_ids_by_subject(self._ids, subjects)

    @staticmethod
    def _check_duplicates(elements: list[Hashable]) -> None:
        num_duplicate = len(elements) - len(set(elements))
        if num_duplicate > 0:
            raise RuntimeError(
                f"Elements must be unique but there are {num_duplicate} duplicate elements!"
            )

    @staticmethod
    def _filter_ids_by_subject(
        ids: list[Identifier], subjects: list[int]
    ) -> "IdentifierSet":
        # Filter
        ids = [f for f in ids if f.subject in set(subjects)]
        # Check for missing subjects
        num_missing_subjects: int = len(subjects) - len(set(f.subject for f in ids))
        if num_missing_subjects > 0:
            print(
                f"IdentifierSet.filter_by_subject(): Out of the {len(subjects)} subjects in  "
                f"{num_missing_subjects} were not found among the given identifiers."
            )
        return IdentifierSet(ids)


class BiometricDataset(abc.ABC):
    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> object:
        identifier = self.ids[index]
        return (identifier.subject, identifier.impression, self.get(identifier))

    def __str__(self) -> str:
        return (
            f"{type(self).__name__} with {self.num_subjects} "
            f"subjects and a total of {len(self)} samples."
        )

    @property
    def num_subjects(self) -> int:
        return self.ids.num_subjects

    @abc.abstractproperty
    def ids(self) -> IdentifierSet:
        pass

    @abc.abstractmethod
    def get(self, identifier: Identifier) -> object:
        pass


class DummyDataset(BiometricDataset):
    def __init__(self, ids: IdentifierSet, value: object):
        self._ids: IdentifierSet = ids
        self._value: object = value

    @property
    def ids(self) -> list[Identifier]:
        return self._ids

    def __len__(self) -> int:
        return len(self._ids)

    def get(self, identifier: Identifier) -> object:
        return self._value


class FilteredDataset(BiometricDataset):
    def __init__(self, dataset: BiometricDataset, identifier_set: IdentifierSet):
        assert identifier_set <= dataset.ids
        self._ids: IdentifierSet = identifier_set
        self._dataset: BiometricDataset = dataset

    @property
    def ids(self) -> IdentifierSet:
        return self._ids

    def get(self, identifier: Identifier) -> object:
        return self._dataset.get(identifier)


class MergedDataset(BiometricDataset):
    def __init__(self, datasets: list[BiometricDataset], share_subjects: bool):
        """
        If share_subjects is set to true it is assumed that the underlying subjects in all datasets are the same.

        i.e. if all Identifiers with subject i accross datasets refer to samples from one finger.

        Otherwise it is assumed that the datasets all refer to disjoint sets of subjects.
        """
        self._datasets: list[BiometricDataset] = datasets
        assert len(self._datasets) > 0

        self._ids = []
        self._id_to_ds: dict[Identifier, int] = {}
        self._id_to_id: dict[Identifier, Identifier] = {}

        if share_subjects:
            impression_count = defaultdict(int)
            for ds_idx, ds in enumerate(self._datasets):
                for id in ds.ids:
                    impression_count[id.subject] += 1
                    new_impression = impression_count[id.subject]
                    impression_count[id.subject] = new_impression
                    new_id = Identifier(id.subject, new_impression)
                    self._ids.append(new_id)
                    self._id_to_ds[new_id] = ds_idx
                    self._id_to_id[new_id] = id
        else:
            subject_count = 0
            subject_map = defaultdict(lambda: None)
            for ds_idx, ds in enumerate(self._datasets):
                for id in ds.ids:
                    new_subject = subject_map[(ds_idx, id.subject)]
                    if new_subject is None:
                        subject_count += 1
                        subject_map[(ds_idx, id.subject)] = subject_count
                        new_subject = subject_count
                    new_id = Identifier(new_subject, id.impression)
                    self._ids.append(new_id)
                    self._id_to_ds[new_id] = ds_idx
                    self._id_to_id[new_id] = id
        self._ids: IdentifierSet = IdentifierSet(self._ids)

    @property
    def ids(self) -> IdentifierSet:
        return self._ids

    def get(self, identifier: Identifier) -> object:
        ds = self._datasets[self._id_to_ds[identifier]]
        return ds.get(self._id_to_id[identifier])


class ZippedDataset(BiometricDataset):
    def __init__(self, datasets: list[BiometricDataset]):
        assert len(datasets) > 0
        self._datasets: list[BiometricDataset] = datasets
        self._ids = datasets[0].ids
        for ds in self._datasets[1:]:
            assert self._ids == ds.ids

    @property
    def ids(self) -> IdentifierSet:
        return self._ids

    def get(self, identifier: Identifier) -> list[object]:
        return [ds.get(identifier) for ds in self._datasets]
