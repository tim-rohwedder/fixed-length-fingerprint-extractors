import os
from dataclasses import dataclass
import random
import json

import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms.functional as VTF


@dataclass
class PoseTransform:
    pad: int = 0  # Pad the image with this amount of white pixels in every direction.
    angle: float = 0  # Rotation in degrees, range [-180, 180]
    shift_horizontal: int = (
        0  # Shift the image this number of pixels to the right; negative value for left
    )
    shift_vertical: int = (
        0  # Shift the image this number of pixels downwards; negative value for upwards
    )

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.pad != 0:
            img = VTF.pad(img, padding=self.pad, fill=1.0)
        if not (
            self.angle == 0 and self.shift_horizontal == 0 and self.shift_vertical == 0
        ):
            img = VTF.affine(
                img,
                angle=self.angle,
                translate=(self.shift_horizontal, self.shift_vertical),
                scale=1,
                shear=0,
                fill=1.0,
            )
        return img

    def transform_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Transfoms coordinates that belong to points in the original image to
        the coordinates of the points in the transformed image.

        @param coordinates : np.ndarray of the coordinates ((x_0, y_0), (x_1, y_1), ...)

        @returns np.ndarray of the transformed coordinates
        """

        # use homogenout coordinates
        homg_coords = np.append(
            coordinates, np.ones(shape=(coordinates.shape[0])), axis=1
        )
        affine_mat = np.array(
            [
                [
                    np.cos(self.angle),
                    -np.sin(self.angle),
                    self.pad + self.shift_horizontal,
                ],
                [
                    np.sin(self.angle),
                    np.cos(self.angle),
                    self.pad + self.shift_vertical,
                ],
                [0, 0, 1],
            ],
            dtype=float,
        )
        transformed_homg = np.matmul(affine_mat, homg_coords)
        transformed_homg = transformed_homg / transformed_homg[:, 2]
        return transformed_homg[:, :2]

    def __str__(self) -> str:
        return f"PoseTransform(angle={self.angle}, shift_horizontal={self.shift_horizontal}, shift_vertical={self.shift_vertical}, pad={self.pad})"


@dataclass
class RandomPoseTransform:
    """
    Resembles a distribution of pose transforms, where angle, vertical and horizontal shift are
    drawn from uniform random distributions. Pad is treated as a constant.
    """

    pad: int = 80
    angle_min: float = -60.0
    angle_max: float = 60.0
    shift_horizontal_min: float = -80.0
    shift_horizontal_max: float = 80.0
    shift_vertical_min: float = -80.0
    shift_vertical_max: float = 80.0

    def sample(self) -> PoseTransform:
        angle = random.uniform(self.angle_min, self.angle_max)
        shift_horizontal = random.uniform(
            self.shift_horizontal_min, self.shift_horizontal_max
        )
        shift_vertical = random.uniform(
            self.shift_vertical_min, self.shift_vertical_max
        )
        return PoseTransform(
            angle=angle,
            shift_vertical=shift_vertical,
            shift_horizontal=shift_horizontal,
            pad=self.pad,
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.sample()(img)

    def save(self, path: str) -> None:
        with open(path, "w") as file:
            json.dump(self.__dict__, file)

    @staticmethod
    def load(path: str) -> "RandomPoseTransform":
        with open(path, "r") as file:
            dist = RandomPoseTransform()
            dist.__dict__ = json.load(file)
            return dist


@dataclass
class RandomQualityTransform:
    gain_min: float = 0.9
    gain_max: float = 1.1
    contrast_min: float = 0.9
    contrast_max: float = 2.0

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        gain = random.uniform(self.gain_min, self.gain_max)
        img = VTF.adjust_brightness(img, brightness_factor=gain)
        contrast = random.uniform(self.contrast_min, self.contrast_max)
        img = VTF.adjust_contrast(img, contrast_factor=contrast)
        return img


def main():
    from flx.visualization.show_with_opencv import show_tensor_as_image
    from flx.setup.paths import get_fingerprint_dataset_path

    img_path = os.path.join(get_fingerprint_dataset_path("SFingev2Example"), "1_1.png")
    img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
    toTensor = torchvision.transforms.ToTensor()
    img: torch.Tensor = toTensor(img)
    show_tensor_as_image(img)

    pose_dist1 = RandomPoseTransform()
    pose_dist1.save("pose_test.json")
    pose_dist2 = RandomPoseTransform.load("pose_test.json")

    quality_trafo = RandomQualityTransform()

    while True:
        pose_trafo = pose_dist2.sample()
        print(pose_trafo)
        transf = pose_trafo(img)

        transf = quality_trafo(transf)
        show_tensor_as_image(transf)


if __name__ == "__main__":
    main()
