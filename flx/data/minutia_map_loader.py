import os
from abc import abstractmethod

import numpy as np
import torch
import torchvision.transforms.functional as VTF

from flx.data.dataset import Identifier, DataLoader, IdentifierSet
from flx.data.minutia_map import create_minutia_map
from flx.data.file_index import FileIndex
from flx.data.iso_encoder_decoder import decode

from flx.data.image_helpers import (
    get_input_resolution,
    transform_to_input_size,
)

"""
The size of the original images created by SFinge.
The minutia locations in the .ist files refer to the original image,
even if the images are resampled to a smaller resolution during generation.
Therefore we need to transform the minutia locations w.r.t. the original
resolution and not w.r.t. the resolution of the actual image.
"""

MINUTIA_MAP_CHANNELS = 6
MINUTIA_MAP_SIZE = 128


class MinutiaMapLoader(DataLoader):
    @abstractmethod
    def get_minutiae(self, identifier: Identifier) -> tuple[np.ndarray, np.ndarray]:
        """
        @returns : Two numpy arrays of equal length. The first array is a 2-D array where each row is
            one coordinate (x, y) of a minutia. The second array is a 1-D array containing the orientation
            in radians (values in range [0.0, 2 * pi]).
        """
        pass

    def get(self, identifier: Identifier) -> tuple[torch.Tensor, float]:
        """
        @returns : Minutia map generated from the minutia points in the given file
        """
        minutiae_loc, minutiae_ori = self.get_minutiae(identifier)
        # Minutiae are often extracted from a different resolution than
        # that of the resized images that serve as input to DeepPrint
        minu_map = create_minutia_map(
            minutiae_loc,
            minutiae_ori,
            in_resolution=get_input_resolution(),
            out_resolution=(MINUTIA_MAP_SIZE, MINUTIA_MAP_SIZE),
            n_layers=MINUTIA_MAP_CHANNELS,
            sigma=1.5,
        )
        return (VTF.to_tensor(minu_map), 1.0)


class SFingeMinutiaMapLoader(MinutiaMapLoader):
    def __init__(
        self,
        root_dir: str,
    ):
        def file_to_id_fun(_: str, filename: str) -> Identifier:
            # Pattern: <dir>/<subject_id>_<impression_id>.png
            filename, _ = os.path.splitext(filename)  # Remove extension(s)
            subject_id, impression_id = filename.split("_")
            # We must start indexing at 0 instead of 1 to be compatible with pytorch
            return Identifier(int(subject_id) - 1, int(impression_id) - 1)

        self._files = FileIndex(root_dir, ".ist", file_to_id_fun)

    @property
    def ids(self) -> IdentifierSet:
        return self._files.ids

    def get_minutiae(self, identifier: Identifier) -> tuple[np.ndarray, np.ndarray]:
        """
        @returns : Two numpy arrays of equal length. The first array is a 2-D array where each row is
            one coordinate (x, y) of a minutia. The second array is a 1-D array containing the orientation
            in radians (values in range [0.0, 2 * pi]).
        """
        minutiae = decode.load_iso19794(
            self._files.get(identifier), format="19794-2-2005"
        )[1:]
        locs = np.array([(m[1], m[2]) for m in minutiae])
        oris = np.array([m[3] for m in minutiae])
        return transform_to_input_size(locs, 560, 416), oris


def _read_mnt_file(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    with open(filepath, "r") as file:
        lines = file.readlines()
        locs = []
        oris = []
        for line in lines[2:]:
            x, y, ori = line.split(" ")
            locs.append((float(x), float(y)))
            oris.append(float(ori))
    return np.array(locs, dtype=np.float16), np.array(oris, dtype=np.float16)


class MCYTOpticalMinutiaMapLoader(MinutiaMapLoader):
    def __init__(
        self,
        root_dir: str,
    ):
        def file_to_id_fun(_: str, filename: str) -> Identifier:
            # Pattern: <dir>/<person:04>_<finger>_<impression>.mnt
            _, person, finger, impression = filename.split("_")
            # 12 impressions per finger
            subject = (10 * int(person)) + int(finger)
            return Identifier(subject, int(impression), finger=int(finger))

        self._files: FileIndex = FileIndex(root_dir, ".mnt", file_to_id_fun)

    @property
    def ids(self) -> IdentifierSet:
        return self._files.ids

    def get_minutiae(self, identifier: Identifier) -> tuple[np.ndarray, np.ndarray]:
        """
        @returns : Two numpy arrays of equal length. The first array is a 2-D array where each row is
            one coordinate (x, y) of a minutia. The second array is a 1-D array containing the orientation
            in radians (values in range [0.0, 2 * pi]).
        """
        locs, oris = _read_mnt_file(self._files.get(identifier))
        return (
            transform_to_input_size(
                locs, original_height=400, original_width=256, roi=(310, 256)
            ),
            oris,
        )


class MCYTCapacitiveMinutiaMapLoader(MinutiaMapLoader):
    def __init__(
        self,
        root_dir: str,
    ):
        def file_to_id_fun(_: str, filename: str) -> Identifier:
            # Pattern: <dir>/<person:04>_<finger>_<impression>.bmp.mnt
            filename = filename[: len(filename) - len(".bmp")]
            _, person, finger, impression = filename.split("_")
            # 12 impressions per finger
            subject = (10 * int(person)) + int(finger)
            return Identifier(subject, int(impression), finger=int(finger))

        self._files: FileIndex = FileIndex(root_dir, ".mnt", file_to_id_fun)

    @property
    def ids(self) -> IdentifierSet:
        return self._files.ids

    def get_minutiae(self, identifier: Identifier) -> tuple[np.ndarray, np.ndarray]:
        """
        @returns : Two numpy arrays of equal length. The first array is a 2-D array where each row is
            one coordinate (x, y) of a minutia. The second array is a 1-D array containing the orientation
            in radians (values in range [0.0, 2 * pi]).
        """
        locs, oris = _read_mnt_file(self._files.get(identifier))
        return transform_to_input_size(locs, 300, 300), oris
