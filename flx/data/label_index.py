from copy import deepcopy

from flx.data.dataset import Identifier, IdentifierSet, DataLoader


class LabelIndex(DataLoader):
    """
    Defines a mapping between subject ids and labels.
    Subjects ids are not continuous while labels are always in range(0, num_subjects)
    """

    def __init__(self, ids: IdentifierSet):
        assert isinstance(ids, IdentifierSet)
        self._ids: IdentifierSet = ids
        subject_set = set(f.subject for f in self._ids)
        self._subject_to_label: dict[int, int] = {
            subject: i for i, subject in enumerate(sorted(subject_set))
        }

    @property
    def ids(self) -> IdentifierSet:
        return self._ids

    def get(self, identifier: Identifier) -> int:
        return self._subject_to_label[identifier.subject]
