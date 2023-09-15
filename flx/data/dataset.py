from typing import Iterable, Hashable
from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict


class Identifier:
    """
    An identifier for data related to a fingerprint sample.
        subject: An identifier for a specific fingerprint
        impression: An identifier for a sample taken of a fingerprint. E.g. if there are three different samples
            of the same fingerprint they all have the same value for 'subject' but different values for 'impression'
    """

    def __init__(self, subject: int, impression: int):
        self.subject: int = int(subject)
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
        return set(self.identifiers) <= set(other.identifiers)

    def __eq__(self, other: "IdentifierSet") -> bool:
        return set(self.identifiers) == set(other.identifiers)

    @property
    def identifiers(self) -> list[Identifier]:
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


class DataLoader(ABC):
    @abstractmethod
    def get(self, identifier: Identifier) -> object:
        pass


class ConstantDataLoader(DataLoader):
    """
    For any identifier, returns the same value.
    """

    def __init__(self, value: object):
        self.value: object = value

    def get(self, identifier: Identifier) -> object:
        return self.value


class ZippedDataLoader(DataLoader):
    """
    For an identifier, the ZippedDataLoader will return a tuple of the results of the individual data loaders.

    The individual data loaders should support the same identifiers, or at least the subset of identifiers which is accessed.
    """

    def __init__(self, loaders: list[DataLoader]):
        assert len(loaders) > 0
        self._loaders: list[DataLoader] = loaders

    def get(self, identifier: Identifier) -> list[object]:
        return [ds.get(identifier) for ds in self._loaders]


class ConcatenatedDataLoader(DataLoader):
    """
    Concatenates multiple data loaders into one. The idea is to have on (external) set of Identifiers and a one-to-one mapping to
    the Identifiers of the individual data loaders.

    Lookup in the concatenated data loader works as follows:
        - We first determine which data loader to use
        - Then we determine the identifier for this specific data loader
        - Then we use the data loader to get the data

    As input we need the mappings:
        External identifier -> data loader
        External identifier -> internal identifier

    """

    def __init__(
        self,
        id_to_loader: dict[Identifier, DataLoader],
        id_to_id: dict[Identifier, Identifier],
    ):
        self._id_to_id: dict[Identifier, Identifier] = id_to_id
        self._id_to_loader: dict[Identifier, DataLoader] = id_to_loader

    def get(self, identifier: Identifier) -> object:
        return self._id_to_loader[identifier].get(self._id_to_id[identifier])


class Dataset:
    """
    A set of identifiers and a data loader that can be used to load the data for these identifiers.

    Allows efficient set operations on the datasets.
    """

    def __init__(self, data_loader: DataLoader, identifier_set: IdentifierSet):
        if not isinstance(identifier_set, IdentifierSet):
            raise ValueError(
                "The identifier_set must be an instance of IdentifierSet!"
            )
        if not isinstance(data_loader, DataLoader):
            raise ValueError("The data_loader must be an instance of DataLoader!")
        
        self.identifier_set = identifier_set
        self.data_loader = data_loader

        if hasattr(data_loader, "ids") and not identifier_set <= data_loader.ids:
            raise ValueError(
                "The identifiers in the dataset must be a subset of the identifiers in the data loader!"
            )

    def __len__(self) -> int:
        return len(self.identifier_set)

    @property
    def ids(self) -> IdentifierSet:
        return self.identifier_set

    @property
    def num_subjects(self) -> int:
        return self.identifier_set.num_subjects

    def __getitem__(self, index: int) -> object:
        identifier = self.identifier_set[index]
        return self.get(identifier)

    def __str__(self) -> str:
        return (
            f"{type(self).__name__} with {self.num_subjects} "
            f"subjects and a total of {len(self)} samples."
        )

    def get(self, identifier: Identifier) -> object:
        return self.data_loader.get(identifier)

    @staticmethod
    def concatenate(*datasets: list["Dataset"], share_subjects: bool = True) -> "Dataset":
        """
        If share_subjects is set to true it is assumed that the underlying subjects in all datasets are the same.
        e.g. subject 1 in the first dataset is exactly the same subject as subject 1 in the second dataset.

        Otherwise it is assumed that the datasets all refer to disjoint sets of subjects.
        e.g. subject 1 in the first dataset is a different subject than subject 1 in the second dataset.
        """
        assert len(datasets) > 0

        ids = []
        id_to_loader: dict[Identifier, int] = {}
        id_to_id: dict[Identifier, Identifier] = {}

        if share_subjects:
            impression_count = defaultdict(int)
            for ds in datasets:
                for id in ds.ids:
                    new_impression = impression_count[id.subject]
                    impression_count[id.subject] += 1
                    new_id = Identifier(id.subject, new_impression)
                    ids.append(new_id)
                    id_to_loader[new_id] = ds.data_loader
                    id_to_id[new_id] = id
        else:
            subject_count = 0
            subject_map = defaultdict(lambda: None)
            for ds_idx, ds in enumerate(datasets):
                for id in ds.ids:
                    new_subject = subject_map[(ds_idx, id.subject)]
                    if new_subject is None:
                        subject_map[(ds_idx, id.subject)] = subject_count
                        new_subject = subject_count
                        subject_count += 1
                    new_id = Identifier(new_subject, id.impression)
                    ids.append(new_id)
                    id_to_loader[new_id] = ds.data_loader
                    id_to_id[new_id] = id
        return Dataset(
            ConcatenatedDataLoader(id_to_loader, id_to_id), IdentifierSet(ids)
        )

    @staticmethod
    def zip(*datasets: list["Dataset"]) -> "Dataset":
        """
        Combines multiple datasets into one by zipping their data loader return values.
        """
        assert len(datasets) > 0
        for ds in datasets[1:]:
            assert datasets[0].ids == ds.ids
        return Dataset(
            ZippedDataLoader([ds.data_loader for ds in datasets]),
            datasets[0].ids,
        )
