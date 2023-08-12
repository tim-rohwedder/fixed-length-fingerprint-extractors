import math

import torchvision.transforms.functional as VTF
import torch

from flx.visualization.show_with_opencv import (
    save_3Dtensor_as_image_grid,
    save_2Dtensor_as_image,
    show_tensor_as_image,
)

from flx.models.torch_helpers import get_device


def _is_even(n: int) -> bool:
    return n % 2 == 0


def _round_uneven(n: int) -> int:
    n = int(n)
    if _is_even(n):
        return n + 1
    return n


def _reshape_first_dim_1(img: torch.Tensor) -> torch.Tensor:
    if len(img.shape) == 2:
        return torch.reshape(img, shape=(1, img.shape[0], img.shape[1]))
    return img


def _make_wave_pattern_scaled(ridge_width: float, n_ridges: int) -> torch.Tensor:
    assert type(n_ridges) == int
    assert n_ridges >= 1

    range_size = n_ridges * torch.pi / 2
    x = torch.linspace(
        -range_size - (torch.pi / 2),
        range_size - (torch.pi / 2),
        _round_uneven(ridge_width * n_ridges),
    )
    x = _reshape_first_dim_1(x.repeat(x.shape[0], 1))
    x = torch.sin(x)
    return x


def _make_gaussian_kernel_2d(
    sigma_x: float, sigma_y: float, size_x: int, size_y: int
) -> torch.Tensor:
    support_y = torch.arange(0, size_y, dtype=torch.float) - (size_y / 2)
    kernel_y = torch.exp(
        torch.distributions.Normal(loc=0, scale=sigma_y).log_prob(support_y)
    )
    support_x = torch.arange(0, size_x, dtype=torch.float) - (size_x / 2)
    kernel_x = torch.exp(
        torch.distributions.Normal(loc=0, scale=sigma_x).log_prob(support_x)
    )
    kernel2d = torch.outer(kernel_y, kernel_x)
    return _reshape_first_dim_1(kernel2d / torch.sum(kernel2d))


@torch.no_grad()
def _make_rotated_filters(
    num_levels: int, ridge_width: float, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates rotated versions of the given ridge filter.
    """
    n_ridges: int = size * 2 + 1
    # Image must be larger by factor of at least 1.41 to show pattern and not fill color in corners when rotated
    n_ridges_full_pattern = _round_uneven(int(n_ridges * 1.41))
    pattern = _make_wave_pattern_scaled(ridge_width, n_ridges_full_pattern)
    gauss = _make_gaussian_kernel_2d(
        sigma_x=pattern.shape[1] * 0.12,
        sigma_y=pattern.shape[2] * 0.24,
        size_x=pattern.shape[1],
        size_y=pattern.shape[2],
    )

    outsize = _round_uneven(n_ridges * ridge_width)
    pattern_rot = []
    angle_per_level = 180.0 / num_levels
    for i in range(num_levels):
        gr = VTF.center_crop(
            VTF.rotate(
                gauss, angle_per_level * i, interpolation=VTF.InterpolationMode.BILINEAR
            ),
            output_size=[outsize, outsize],
        )
        gr = gr / torch.sum(gr)

        pr = VTF.center_crop(
            VTF.rotate(
                pattern,
                angle_per_level * i,
                interpolation=VTF.InterpolationMode.BILINEAR,
            ),
            output_size=[outsize, outsize],
        )
        pr = pr * gr
        pr = pr - (torch.sum(pr) / pr.numel())
        pr = pr / torch.sum(torch.abs(pr))
        pattern_rot.append(pr)
    pattern_rot = torch.concat(pattern_rot)
    pattern_rot = torch.reshape(pattern_rot, shape=(num_levels, 1, outsize, outsize))
    return pattern_rot


def _normalize_0_1(img: torch.Tensor) -> torch.Tensor:
    img_min = torch.min(img)
    img_max = torch.max(img)
    if not img_max > img_min:
        return torch.zeros_like(img)
    return (img - img_min) / (img_max - img_min)


class _GaborFilter:
    def __init__(self, ridge_width: float):
        self._pattern = _make_rotated_filters(16, ridge_width, 2)

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img_filtered = torch.nn.functional.conv2d(
            img, self._pattern, stride=1, padding="valid"
        )

        ridges = -torch.amin(img_filtered, dim=0)
        ridges = torch.threshold(ridges, 0.0, 0.0)
        ridges = _reshape_first_dim_1(ridges)
        assert torch.amin(0.0 <= ridges)
        assert torch.amax(ridges <= 1.0)
        size_diff = img.shape[-1] - ridges.shape[-1]
        assert size_diff % 2 == 0
        return VTF.pad(ridges, padding=int(size_diff / 2), fill=0.0)


def _pad_to_match_shape(img: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    x_diff = other.shape[-1] - img.shape[-1]
    assert x_diff > 0
    y_diff = other.shape[-2] - img.shape[-2]
    assert y_diff > 0
    pad_left = int(x_diff / 2)
    pad_right = x_diff - pad_left
    pad_top = int(y_diff / 2)
    pad_bot = y_diff - pad_top
    return VTF.pad(img, [pad_left, pad_top, pad_right, pad_bot], 0.0)


class _FingerprintBinarizer:
    SMOOTHING_FACTOR = 0.6

    def __init__(self, ridge_width: float):
        self._gabor_filters = [
            _GaborFilter(ridge_width * 0.6),
            _GaborFilter(ridge_width),
            _GaborFilter(ridge_width / 0.6),
        ]

        self._blur_kernel_size: int = _round_uneven(ridge_width)
        self._blur_kernel_sigma: float = self._blur_kernel_size * self.SMOOTHING_FACTOR

    @torch.no_grad()
    def binarize(self, img: torch.Tensor) -> torch.Tensor:
        img = _normalize_0_1(img)
        img_inv = 1.0 - img
        # show_tensor_as_image(img, "IN", wait=False)

        ridges = self._gabor_filters[0](img)
        valleys = self._gabor_filters[0](-img)
        for f in self._gabor_filters[1:]:
            ridges = torch.maximum(ridges, f(img))
            valleys = torch.maximum(valleys, f(img_inv))
        ridges = _normalize_0_1(ridges)
        valleys = _normalize_0_1(valleys)

        # show_tensor_as_image(ridges, "ridges", wait=False)
        # show_tensor_as_image(valleys, "valleys", wait=False)

        segmented = VTF.gaussian_blur(
            ridges + valleys, self._blur_kernel_size * 7, self._blur_kernel_sigma * 7
        )
        segmented = (segmented > 0.1).float()
        # Ignore areas at the edge, they are not reliable
        segmented = VTF.resize(
            segmented,
            [int(segmented.shape[-2] * 0.9), int(segmented.shape[-1] * 0.9)],
            antialias=True,
        )
        segmented = _pad_to_match_shape(segmented, ridges)
        # show_tensor_as_image(segmented, "segmented", wait=False)

        ridges = VTF.adjust_contrast(ridges - valleys, 8.0) * segmented
        # show_tensor_as_image(ridges, "ridges minus valleys after seg", wait=False)

        ridges = (ridges > 0.2).float()
        # show_tensor_as_image(ridges, "OUT")

        return ridges


class LazilyAllocatedBinarizer:
    def __init__(self, ridge_width: float):
        self._ridge_width: float = ridge_width
        self._binarizer: _FingerprintBinarizer = None

    def __call__(self, img: torch.Tensor):
        if self._binarizer is None:
            self._binarizer = _FingerprintBinarizer(self._ridge_width)
        return self._binarizer.binarize(img)
