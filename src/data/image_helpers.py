from typing import Union

import numpy as np
import torch
import torchvision.transforms.functional as VTF

from src.setup.config import INPUT_SIZE


def get_deepfinger_input_resolution() -> tuple[int, int]:
    return (INPUT_SIZE, INPUT_SIZE)


def pad_and_resize(
    img: Union[np.ndarray, torch.Tensor],
    target_size: tuple[int, int] = None,
    fill: float = 0.0,
) -> torch.Tensor:
    if not isinstance(img, torch.Tensor):
        img = VTF.to_tensor(img)

    height = img.shape[1]
    width = img.shape[2]
    pad_width = 0 if width >= height else int((height - width) / 2)
    pad_height = 0 if height >= width else int((width - height) / 2)
    img = VTF.pad(
        img, padding=(pad_width, pad_height, pad_width, pad_height), fill=fill
    )  # left, top, right, bottom

    assert img.shape[1] == img.shape[2]

    return VTF.resize(img, target_size, antialias=True)


def pad_and_resize_to_deepfinger_input_size(
    img: Union[np.ndarray, torch.Tensor],
    roi: Union[None, tuple[int, int]] = None,
    fill: float = 0.0,
) -> torch.Tensor:
    if not isinstance(img, torch.Tensor):
        img = VTF.to_tensor(img)

    if roi is not None:
        img = VTF.center_crop(img, roi)

    return pad_and_resize(
        img, (INPUT_SIZE, INPUT_SIZE), fill=fill
    )


def transform_to_deepfinger_input_size(
    minutia_points: np.ndarray,
    original_height: int,
    original_width: int,
    roi: Union[None, tuple[int, int]] = None,
) -> np.ndarray:
    """
    Transforms the pixel coordinates in the same way that the pixels in the original image would be
    transformed by pad_and_resize_to_deepfinger_input_size.
    """
    minutia_points = minutia_points.astype(np.float16)
    
    if minutia_points.shape[0] == 0:
        return minutia_points

    if roi is not None:
        minutia_points -= np.array(
            [(original_width - roi[1]) / 2, (original_height - roi[0]) / 2]
        )
        height = roi[0]
        width = roi[1]
    else:
        height = original_height
        width = original_width

    pad_width = 0 if width >= height else (height - width) / 2
    pad_height = 0 if height >= width else (width - height) / 2

    minutia_points += np.array([pad_width, pad_height])
    minutia_points *= INPUT_SIZE / max(height, width)
    return minutia_points
