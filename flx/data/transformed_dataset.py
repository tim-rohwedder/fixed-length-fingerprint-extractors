from typing import Union, Callable

import torch

from flx.data.fingerprint_dataset import FingerprintDataset
from flx.data.biometric_dataset import Identifier, BiometricDataset
from flx.data.pose_dataset import PoseDataset
from flx.preprocessing.augmentation import (
    RandomPoseTransform,
)


class TransformedDataset(BiometricDataset):
    def __init__(
        self,
        images: FingerprintDataset,
        poses: Union[None, PoseDataset, RandomPoseTransform] = None,
        transforms: list[Callable[[torch.Tensor], torch.Tensor]] = [],
    ):
        """
        A lightweight wrapper for applying image augmentation to an existing fingerprint dataset.

        First, if 'augment_image_pose' is True, a pose transform can be applied, it consists of:
            - Padding
            - Rotation
            - Shift in x and y direction
        If 'poses' is specified the rotation and shift are taken from this FingerprintPoseDataset,
        otherwise they are randomly generated.

        Then, if 'augment_image_quality' is True, the following properties are randomly transformed:
            - Gain (aka brightness)
            - Contrast
            - Gamma

        """
        self._fingerprints: FingerprintDataset = images
        self._pose_distribution: RandomPoseTransform = None
        self._pose_databset: PoseDataset = None
        self._transforms: list[Callable[[torch.Tensor], torch.Tensor]] = transforms

        if not (
            poses is None
            or isinstance(poses, PoseDataset)
            or isinstance(poses, RandomPoseTransform)
        ):
            raise TypeError(
                "Parameter 'poses' must be None, a PoseDataset or a PoseDistribution"
            )

        if isinstance(poses, PoseDataset):
            if not set(images.ids).issubset(set(poses.ids)):
                raise RuntimeError(
                    "AugmentedFingerprints received an incomplete pose dataset"
                )
            self._pose_databset = poses

        if isinstance(poses, RandomPoseTransform):
            self._pose_distribution = poses

    @property
    def ids(self) -> list[Identifier]:
        return self._fingerprints.ids

    def get(self, identifier: Identifier) -> torch.Tensor:
        """
        Get a fingerprint sample from the underlying dataset with the described augmentations applied.

        The transforms are only applied to the fingerprint image, not the minutia map.

        @returns tuple of fingerprint image, minutia map, label, subject, impression
        """
        img = self._fingerprints.get(identifier)

        if self._pose_databset is not None:
            img = self._pose_databset.get(identifier)(img)
        elif self._pose_distribution is not None:
            img = self._pose_distribution.sample()(img)

        for transform in self._transforms:
            img = transform(img)

        return img
