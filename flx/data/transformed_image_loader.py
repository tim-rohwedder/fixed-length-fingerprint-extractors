from typing import Union, Callable

import torch

from flx.data.dataset import Identifier, IdentifierSet, DataLoader
from flx.data.image_loader import ImageLoader
from flx.data.pose_dataset import PoseLoader
from flx.image_processing.augmentation import (
    RandomPoseTransform,
)


class TransformedImageLoader(DataLoader):
    def __init__(
        self,
        images: ImageLoader,
        poses: Union[None, PoseLoader, RandomPoseTransform] = None,
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
        self._images: ImageLoader = images
        self._pose_distribution: RandomPoseTransform = None
        self._pose_dataset: PoseLoader = None
        self._transforms: list[Callable[[torch.Tensor], torch.Tensor]] = transforms

        if not (
            poses is None
            or isinstance(poses, PoseLoader)
            or isinstance(poses, RandomPoseTransform)
        ):
            raise TypeError(
                "Parameter 'poses' must be None, a PoseDataset or a PoseDistribution"
            )

        if isinstance(poses, PoseLoader):
            if not set(images.ids).issubset(set(poses.ids)):
                raise RuntimeError("Received an incomplete pose dataset")
            self._pose_dataset = poses

        if isinstance(poses, RandomPoseTransform):
            self._pose_distribution = poses

    @property
    def ids(self) -> IdentifierSet:
        return self._images.ids

    def get(self, identifier: Identifier) -> torch.Tensor:
        """
        Get a fingerprint sample from the underlying dataset with the described augmentations applied.

        The transforms are only applied to the fingerprint image, not the minutia map.

        @returns tuple of fingerprint image, minutia map, label, subject, impression
        """
        img = self._images.get(identifier)

        if self._pose_dataset is not None:
            img = self._pose_dataset.get(identifier)(img)
        elif self._pose_distribution is not None:
            img = self._pose_distribution.sample()(img)

        for transform in self._transforms:
            img = transform(img)

        return img
