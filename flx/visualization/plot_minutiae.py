import cv2
import numpy as np
import torch
from typing import List, Tuple

from flx.visualization.show_with_opencv import (
    _tensor_to_2Dnumpy_array,
    _normalized_array_to_grayscale,
)

# Define the constants for the colors and circle/line parameters
MINUTIA_DOT_COLOR = (255, 0, 0)  # Blue in BGR format
MINUTIA_DOT_RADIUS = 5  # Pixels
LINE_COLOR = (0, 255, 0)  # Green in BGR format
LINE_THICKNESS = 2  # Pixels
LINE_LENGTH = 8  # Pixels


def _plot_points_and_lines(
    img: cv2.Mat, coords: list[list[int]], angles: list[float], out_path: str
) -> None:
    # Read the image from the filepath

    # Loop through the coordinates and angles arrays
    for i in range(len(coords)):
        # Get the current point and angle
        x, y = coords[i]
        angle = angles[i]

        # Draw a circle at the point with the defined color and radius
        cv2.circle(img, (x, y), MINUTIA_DOT_RADIUS, MINUTIA_DOT_COLOR, -1)

        # Calculate the end point of the line using trigonometry
        x2 = int(x + LINE_LENGTH * np.cos(angle))
        y2 = int(y + LINE_LENGTH * np.sin(angle))

        # Draw a line from the point to the end point with the defined color and thickness
        cv2.line(img, (x, y), (x2, y2), LINE_COLOR, LINE_THICKNESS)

    # Save the updated image to the output filepath
    cv2.imwrite(out_path, img)


def plot_minutiae(
    out_path: str, image: cv2.Mat, locs: np.ndarray, oris: np.ndarray
) -> None:
    if isinstance(image, torch.Tensor):
        image = _normalized_array_to_grayscale(_tensor_to_2Dnumpy_array(image))

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    locs = locs.astype(int)
    _plot_points_and_lines(image, locs, oris, out_path)
