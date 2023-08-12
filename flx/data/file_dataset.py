from typing import Generator, Callable
import os

from flx.data.biometric_dataset import Identifier, IdentifierSet, BiometricDataset


def _get_subdirs_with_files(
    root_dir: str, extension: str
) -> Generator[tuple[str, list[str]], None, None]:
    """
    Yiels tuple of relative path from root to subdirectory and filenames in subdirectory
    for every subdirectory that contains files.
    """
    extension = extension.lower()
    for dir, _, files in os.walk(root_dir):
        files = [f for f in files if os.path.splitext(f)[1].lower() == extension]
        if len(files) == 0:
            continue
        relative_path_from_root = os.path.relpath(dir, root_dir)
        yield (relative_path_from_root, files)


class FileDataset(BiometricDataset):
    ELEMENT_TYPE = str

    def __init__(
        self,
        root_dir: str,
        file_extension: str,
        file_to_id: Callable[[str], Identifier],
        id_to_file: Callable[[Identifier], str],
    ):
        """
        Dataset which contains fingerprint images in subdirectories.
        Discovers all files with the given extension in the directory and
        handles mapping between paths and ids. The mapping function between filename
        and the id and the image loading has to be implemented in the deriving function.

        Folder structure:
        <root_dir>/
            <subdir>/
                ...
                    <subdir>/
                        <image>
                        <image>
                        ...
                    <subdir>/
                        ...
        """
        self._root_dir: str = root_dir
        self._file_extension: str = file_extension
        self._file_to_id: Callable[[str], Identifier] = file_to_id
        self._id_to_file: Callable[[Identifier], str] = id_to_file

        self._id_to_subdir: dict = dict()
        self._ids = []
        for subdir, files in _get_subdirs_with_files(
            self._root_dir, self._file_extension
        ):
            for f in files:
                identifier = self._file_to_id(f)
                self._id_to_subdir[identifier] = subdir
                self._ids.append(identifier)
        self._ids: IdentifierSet = IdentifierSet(self._ids)
        if len(self) == 0:
            print(
                f"FileDataset with root_dir '{self._root_dir}' is empty. No files with extension {self._file_extension} found."
            )

    @property
    def ids(self) -> IdentifierSet:
        return self._ids

    def get(self, identifier: Identifier) -> str:
        path = os.path.join(
            self._root_dir,
            self._id_to_subdir[identifier],
            self._id_to_file(identifier),
        )
        assert os.path.exists(path)
        return path
