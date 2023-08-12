import os
import abc

import numpy as np
from PIL import Image
import wsq

# import wsq  # needed for loading nist sd 14 dataset
import torch
import torchvision.transforms.functional as VTF
import cv2

from flx.data.image_helpers import (
    pad_and_resize_to_deepfinger_input_size,
)
from flx.data.biometric_dataset import Identifier, IdentifierSet, BiometricDataset
from flx.data.file_dataset import FileDataset


class FingerprintDataset(BiometricDataset):
    def __init__(self, root_dir: str, file_extension: str):
        self._files = FileDataset(
            root_dir, file_extension, self._file_to_id_fun, self._id_to_file_fun
        )

    @property
    def ids(self) -> IdentifierSet:
        return self._files.ids

    @abc.abstractstaticmethod
    def _file_to_id_fun(filename: str) -> Identifier:
        pass

    @abc.abstractstaticmethod
    def _id_to_file_fun(identifier: Identifier) -> str:
        pass


class SFingeDataset(FingerprintDataset):
    def __init__(
        self,
        root_dir: str,
    ):
        super(SFingeDataset, self).__init__(root_dir, ".png")

    @staticmethod
    def _file_to_id_fun(filename: str) -> Identifier:
        # Pattern: <dir>/<subject_id>_<impression_id>.png
        filename, _ = os.path.splitext(filename)  # Remove extension(s)
        subject_id, impression_id = filename.split("_")
        # We must start indexing at 0 instead of 1 to be compatible with pytorch
        return Identifier(int(subject_id) - 1, int(impression_id) - 1)

    @staticmethod
    def _id_to_file_fun(identifier: Identifier) -> str:
        return f"{identifier.subject + 1}_{identifier.impression + 1}.png"

    def get(self, identifier: Identifier) -> torch.Tensor:
        img = cv2.imread(self._files.get(identifier), flags=cv2.IMREAD_GRAYSCALE)
        return VTF.to_tensor(img[:-32])


class FVC2004Dataset(FingerprintDataset):
    def __init__(
        self,
        root_dir: str,
    ):
        super(FVC2004Dataset, self).__init__(root_dir, ".tif")

    @staticmethod
    def _file_to_id_fun(filename: str) -> Identifier:
        # Pattern: <dir>/<subject_id>_<sample_id>.png
        filename, _ = os.path.splitext(filename)  # Remove extension(s)
        subject_id, impression_id = filename.split("_")
        # We must start indexing at 0 instead of 1 to be compatible with pytorch
        return Identifier(int(subject_id) - 1, int(impression_id) - 1)

    @staticmethod
    def _id_to_file_fun(identifier: Identifier) -> cv2.Mat:
        return f"{identifier.subject + 1}_{identifier.impression + 1}.tif"

    def get(self, identifier: Identifier) -> torch.Tensor:
        img = cv2.imread(self._files.get(identifier), cv2.IMREAD_GRAYSCALE)
        return pad_and_resize_to_deepfinger_input_size(img, fill=1.0)


class MCYTOpticalDataset(FingerprintDataset):
    def __init__(
        self,
        root_dir: str,
    ):
        super(MCYTOpticalDataset, self).__init__(root_dir, ".bmp")

    @staticmethod
    def _file_to_id_fun(filename: str) -> Identifier:
        # Pattern: <dir>/<person>_<finger>_<impression>.png
        filename, _ = os.path.splitext(filename)  # Remove extension(s)
        _, person, finger, impression = filename.split("_")
        # 12 impressions per finger
        subject = (10 * int(person)) + int(finger)
        return Identifier(subject, int(impression), finger=int(finger))

    @staticmethod
    def _id_to_file_fun(identifier: Identifier) -> cv2.Mat:
        person = int(identifier.subject / 10)
        finger = int(identifier.subject % 10)
        return f"dp_{person:04d}_{finger}_{identifier.impression}.bmp"

    def get(self, identifier: Identifier) -> torch.Tensor:
        img = cv2.imread(self._files.get(identifier), cv2.IMREAD_GRAYSCALE)
        width = img.shape[1]
        height = img.shape[0]
        return pad_and_resize_to_deepfinger_input_size(img, roi=(310, width), fill=1.0)


class MCYTCapacitiveDataset(FingerprintDataset):
    def __init__(
        self,
        root_dir: str,
    ):
        super(MCYTCapacitiveDataset, self).__init__(root_dir, ".bmp")

    @staticmethod
    def _file_to_id_fun(filename: str) -> Identifier:
        # Pattern: <dir>/<person>_<finger>_<impression>.png
        filename, _ = os.path.splitext(filename)  # Remove extension(s)
        _, person, finger, impression = filename.split("_")
        # 12 impressions per finger
        subject = (10 * int(person)) + int(finger)
        return Identifier(subject, int(impression), finger=int(finger))

    @staticmethod
    def _id_to_file_fun(identifier: Identifier) -> cv2.Mat:
        person = int(identifier.subject / 10)
        finger = int(identifier.subject % 10)
        return f"pb_{person:04d}_{finger}_{identifier.impression}.bmp"

    def get(self, identifier: Identifier) -> torch.Tensor:
        img = cv2.imread(self._files.get(identifier), cv2.IMREAD_GRAYSCALE)
        return pad_and_resize_to_deepfinger_input_size(img, fill=1.0)


class NistSD4Dataset(FingerprintDataset):
    def __init__(
        self,
        root_dir: str,
    ):
        super(NistSD4Dataset, self).__init__(root_dir, ".png")

    @staticmethod
    def _file_to_id_fun(filename: str) -> Identifier:
        # Pattern: <dir>/[f|s]<subject>_<finger>.png
        filename, _ = os.path.splitext(filename)  # Remove extension(s)
        sample = 0 if filename[0] == "f" else 1
        subject, finger = filename[1:].split("_")
        # We must start indexing at 0 instead of 1 to be compatible with pytorch
        return Identifier(
            subject=int(subject) - 1,
            impression=sample,
            finger=int(finger),
        )

    @staticmethod
    def _id_to_file_fun(identifier: Identifier) -> str:
        samplechar = "f" if identifier.impression == 0 else "s"
        return f"{samplechar}{identifier.subject + 1:04d}_{identifier.finger:02d}.png"

    def get(self, identifier: Identifier) -> torch.Tensor:
        img = cv2.imread(self._files.get(identifier), cv2.IMREAD_GRAYSCALE)
        return pad_and_resize_to_deepfinger_input_size(img, fill=1.0)


class NistSD14Dataset(FingerprintDataset):
    def __init__(
        self,
        root_dir: str,
    ):
        super(NistSD14Dataset, self).__init__(root_dir, ".WSQ")

    @staticmethod
    def _file_to_id_fun(filename: str) -> Identifier:
        # Pattern: <dir>/[F|S]<subject>.WSQ
        filename = os.path.basename(filename)
        filename, _ = os.path.splitext(filename)  # Remove extension(s)
        sample = 0 if filename[0] == "F" else 1
        subject = int(filename[1:]) - 1
        # We must start indexing at 0 instead of 1 to be compatible with pytorch
        return Identifier(
            subject=subject,
            impression=sample,
        )

    @staticmethod
    def _id_to_file_fun(identifier: Identifier) -> str:
        samplechar = "F" if identifier.impression == 0 else "S"
        return f"{samplechar}{identifier.subject + 1:07d}.WSQ"

    def get(self, identifier: Identifier) -> torch.Tensor:
        img = Image.open(self._files.get(identifier)).convert("L")
        img = np.array(img, dtype=np.uint8)
        return pad_and_resize_to_deepfinger_input_size(img, fill=1.0)
