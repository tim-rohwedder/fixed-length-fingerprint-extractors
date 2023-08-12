import numpy as np


from flx.visualization.show_with_opencv import show_minutia_maps


def _remove_points_outside_image(
    coords: np.ndarray, oris: np.ndarray, height: int, width: int
) -> np.ndarray:
    mask_x = coords[:, 0] >= width
    mask_y = coords[:, 1] >= height
    mask = np.logical_or(mask_x, mask_y)
    coords = np.delete(coords, np.where(mask), axis=0)
    oris = np.delete(oris, np.where(mask))
    return coords, oris


def _rescale_points(
    points: np.ndarray,
    image_resolution: tuple[int, int],
    target_resolution: tuple[int, int],
):
    scale_factor = min(
        target_resolution[0] / image_resolution[0],
        target_resolution[1] / image_resolution[1],
    )

    padding_x = (target_resolution[1] - (image_resolution[1] * scale_factor)) / 2
    padding_y = (target_resolution[0] - (image_resolution[0] * scale_factor)) / 2

    assert padding_x >= 0
    assert padding_y >= 0

    padding = np.array([padding_x, padding_y], dtype=np.uint16)

    return points * scale_factor + padding


def _gaussian_mask(sigma: float, radius: int):
    x, y = np.meshgrid(
        np.linspace(-radius, radius, 2 * radius + 1),
        np.linspace(-radius, radius, 2 * radius + 1),
    )
    dst = x**2 + y**2

    return np.exp(-(dst / (2.0 * sigma**2)))


def _convert_orientations(orientations: np.ndarray[float]) -> np.ndarray:
    """
    Makes sure that the orientations in the output array are in range
    [0, 2 * pi]
    """
    out = orientations.copy()
    two_pi_inv = 1 / (2 * np.pi)
    out *= two_pi_inv
    out = out - np.floor(out)
    out *= 2 * np.pi
    return out


def _layer_weights_softmax(orientations: np.ndarray[float], n_layers: int):
    """
    To calculate the layer weights for a given orientation we first calculate the orientation
    difference to each layer's orientation. The final weight is then obtained by applying
    the softmax function over all layers.

    @returns :
        np.ndarray of type float16 and shape (orientations.shape[0], n_layers)
    """
    layer_orientations = np.linspace(
        0, 2 * np.pi, num=n_layers, endpoint=False, dtype=np.float16
    )
    layer_orientations = np.tile(layer_orientations, (orientations.shape[0], 1))
    orientation_diffs = np.abs(layer_orientations - orientations[:, np.newaxis])

    # If greater than pi -> We calculated the larger of the two angles
    # between the orientations -> Use   2 pi - ori   instead
    mask = orientation_diffs > np.pi
    orientation_diffs[mask] *= -1
    orientation_diffs[mask] += 2 * np.pi

    weights = np.exp(-orientation_diffs)
    norm = 1 / np.sum(weights, axis=1)
    return weights * norm[:, np.newaxis]


def create_minutia_map(
    minutia_locations: np.ndarray,
    minutia_orientations: np.ndarray,
    in_resolution: tuple[int, int],
    out_resolution: tuple[int, int],
    n_layers: int,
    sigma: int,
) -> np.ndarray:
    """
    Creates a minutia map representation of the given minutia points.
    Follows the procedure described in

    End-to-End Latent Fingerprint Search
    Kai Cao, Dinh-Luan Nguyen, Cori Tymoszek, A.K. Jain
    https://arxiv.org/abs/1812.10213v1

    First the points are rescaled to match the output resolution.

    The output matrix is initialized with zeros. Then for each minutia point, the density
    values of a gaussian distribution with the center at the location and a standard deviation of
    'sigma' are added to the layers. For each layer, these densities are further weighted
    according to the minutia orientation.

    For performance, the radius around each minutia point where this is applied is limited
    to 2 * sigma.

    We then rescale the image, so that the density at each minutia point location
    corresponds to a pixel value of 255.

    @param minutia_locations :
        numpy int array of shape (num_minutiae, 2)
        Contains the x, y pixel position of the minutia point. Assumes OpenCV coordinates
        (i.e. x from left to right, y from top to bottom)
        Like [[x1, y1], [x2, y2], [x3, y4], ...]
    @param minutia_orientations :
        numpy float array of shape (num_minutiae)
        Contains the corresponding minutia orientations where e.g.
            > 0.0    is orientation in the direction of the x-Axis (to the right)
            > 1/2 pi is the direction of the y-Axis (upwards)
            > pi (or -pi) is the direction of the negative x-Axis (to the left)
            > 3/2 pi (or - 1/2 pi) is the direction of the negative y-Axis (downwards)
    @param image_resolution :
        resolution of the fingerprint image (width, height)
    @param out_resolution :
        target resolution of the minutia map
        INFO: The resulting minutia map includes padding of size 2 * sigma
        INFO: The shape of the output matrix will be (height, width) not (width, height)
    @param n_layers :
        number of layers; The k-th layer corresponds to the orientation  k * 2 * pi / n_layers
    @param sigma :
        Determines the size of a minutia point in the target map; Higher value means larger minutia points

    @returns :
        np.ndarray of type np.uint8 and shape (out_resolution[1], out_resolution[0], n_layers)
    """
    radius = int(np.ceil(2 * sigma))
    out_image = np.zeros(
        shape=(
            out_resolution[1] + 2 * radius,
            out_resolution[0] + 2 * radius,
            n_layers,
        ),
        dtype=np.float16,
    )

    if minutia_locations.shape[0] == 0:
        return out_image[radius:-radius, radius:-radius].astype(dtype=np.uint8)

    if in_resolution != out_resolution:
        minutia_locations = _rescale_points(
            minutia_locations,
            image_resolution=in_resolution,
            target_resolution=out_resolution,
        ).astype(np.uint16)

    minutia_locations, minutia_orientations = _remove_points_outside_image(
        minutia_locations, minutia_orientations, out_resolution[1], out_resolution[0]
    )

    minu_layer_weights = _layer_weights_softmax(
        _convert_orientations(minutia_orientations), n_layers=n_layers
    )

    mask = _gaussian_mask(sigma, radius)
    base_density = np.reshape(
        np.repeat(mask, n_layers), newshape=(mask.shape[0], mask.shape[0], n_layers)
    )

    for minu_idx, layer_weights in enumerate(minu_layer_weights[:]):
        minu_density = base_density * layer_weights[np.newaxis, np.newaxis, :]

        out_loc = minutia_locations[minu_idx] + radius
        x_start = out_loc[0] - radius
        x_end = out_loc[0] + radius + 1
        y_start = out_loc[1] - radius
        y_end = out_loc[1] + radius + 1
        out_image[y_start:y_end, x_start:x_end, :] += minu_density

    out_image = np.clip(out_image, 0, 1.0)
    out_image *= 255
    return out_image[radius:-radius, radius:-radius].astype(dtype=np.uint8)


def main():
    # Locations: left-top, right-bottom, left-middle, right-top
    locations = np.array([(100, 100), (300, 300), (100, 200), (350, 50)])
    orientations = np.array([0.0, np.pi, 3 / 2 * np.pi, -1.0])
    image_resolution = (400, 400)
    output_resolution = (128, 128)
    maps = create_minutia_map(
        locations,
        orientations,
        image_resolution,
        output_resolution,
        n_layers=4,
        sigma=1.5,
    )
    print(maps.shape)
    show_minutia_maps(maps)


if __name__ == "__main__":
    main()
