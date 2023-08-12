from typing import Generator, Callable
import os

from flx.data.dataset import Identifier, IdentifierSet, DataLoader


def _get_subdirs_with_files(
    root_dir: str, extension: str
) -> Generator[tuple[str, list[str]], None, None]:
    """
    Yiels tuple of relative path from root to subdirectory and filenames in subdirectory
    for every subdirectory that contains files.
    """
    extension = extension.lower()
    for dir, _, files in os.walk(root_dir):
        files = [os.path.splitext(f) for f in files if os.path.splitext]
        files = [f for f, ext in files if ext.lower() == extension]
        if len(files) == 0:
            continue
        relative_path_from_root = os.path.relpath(dir, root_dir)
        yield (relative_path_from_root, files)


class FileIndex(DataLoader):
    def __init__(
        self,
        root_dir: str,
        file_extension: str,
        id_from_path: Callable[[str, str], Identifier],
    ):
        """
        A DataLoader for filepaths.

        Discovers all files with the given extension in `root_dir` and its subdirectories and
        handles mapping between relative paths and ids. For each path relative to `root_dir` there
        must be a unique id and vice versa.

        file_extension: e.g. ".png"
        file_to_id_fun: function that maps a relative path to an id.
            E.g. if the file path is "~/dataset_dir/person005/finger08/file000.png"
            and the root_dir is "~/dataset_dir", then the function will receive
            ("person005/finger08", "file000") as arguments.
            The id could be Identifier(subject=6*100 + 7, finger=7, sample=0)

        Folder structure:
        root_dir/
            subdir/
                ...
                    subdir/
                        file.extension
                        file.extension
                        ...
                    subdir/
                        ...
        """
        self._root_dir: str = os.path.normpath(os.path.abspath(root_dir))

        self._file_extension: str = file_extension
        if not self._file_extension.startswith("."):
            self._file_extension = "." + self._file_extension

        # Tuple is (subdir, filename) where subdir is relative to root_dir
        self._id_to_path_components: dict[Identifier, tuple[str, str]] = {}

        for subdir, files in _get_subdirs_with_files(
            self._root_dir, self._file_extension
        ):
            for f in files:
                identifier = id_from_path(subdir, f)
                self._id_to_path_components[identifier] = (subdir, f)

        if len(self._id_to_path_components) == 0:
            print(
                f"FileDataset with root_dir '{self._root_dir}' is empty. No files with extension {self._file_extension} found."
            )
        self._ids = IdentifierSet(self._id_to_path_components.keys())

    @property
    def ids(self) -> IdentifierSet:
        return self._ids

    def get(self, identifier: Identifier) -> str:
        subdir, file = self._id_to_path_components[identifier]
        relpath = file if subdir == "." else os.path.join(subdir, file)
        path = os.path.join(self._root_dir, relpath + self._file_extension)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File in index not found: {path}")
        return path
