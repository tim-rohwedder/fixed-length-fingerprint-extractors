from os.path import splitext
import math

import torch
import cv2
import numpy as np

from flx.setup.config import INTERACTIVE_VIS


@torch.no_grad()
def _make_grid(images: list[np.ndarray], ncols: int) -> np.ndarray:
    """
    Orders images of equal size into a grid with ncol columns
    """
    nrows = math.ceil(len(images) / ncols)
    imshape = images[0].shape
    gridshape = (imshape[0] * nrows, imshape[1] * ncols)

    assert gridshape[0] < 4000
    assert gridshape[1] < 4000

    outim = np.zeros(shape=gridshape, dtype=np.uint8)
    imidx = 0
    for i in range(nrows):
        for j in range(ncols):
            if imidx < len(images):
                outim[
                    i * imshape[0] : (i + 1) * imshape[0],
                    j * imshape[1] : (j + 1) * imshape[1],
                ] = images[imidx]
            imidx += 1
    return outim


def _normalized_array_to_grayscale(arr: np.ndarray) -> np.ndarray:
    amin = arr.min()
    amax = arr.max()
    if amin < 0.0 or amax > 1.0 or amax - amin < 0.1:
        if amax - amin != 0:
            arr = (arr - amin) * (1 / (amax - amin))
    arr = arr * 255
    return arr.astype(np.uint8)


@torch.no_grad()
def _tensor_to_2Dnumpy_array(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.to(torch.device("cpu"))
    arr = tensor.numpy()
    if len(arr.shape) != 2:
        new_shape = [s for s in arr.shape if s != 1]
        if len(new_shape) != 2:
            raise RuntimeError(
                f"show_tensor_as_image: Cannot convert tensor with shape {arr.shape} to 2D numpy array!"
            )
        arr = np.resize(arr, new_shape=new_shape)
    return arr


def show_minutia_maps(minu_maps: np.ndarray) -> None:
    """
    @param minu_maps : np.ndarray of type np.uint8 must have shape (height, width, n_layers)
    """
    if not INTERACTIVE_VIS:
        return
    n_layers = minu_maps.shape[2]
    for i in range(n_layers):
        cv2.imshow(
            f"Minutia map {int(i * 360 / n_layers):3} Degrees Orientation (0 is x Axis)",
            minu_maps[:, :, i],
        )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


@torch.no_grad()
def show_minutia_maps_from_tensor(minu_maps: torch.Tensor) -> None:
    """
    @param minu_maps : [0, 1] normalized torch.Tensor; must have shape (n_layers, height, width)
    """
    if not INTERACTIVE_VIS:
        return
    for i, layer in enumerate(minu_maps[:]):
        cv2.imshow(
            f"Minutia map {int(i * 360 / minu_maps.shape[0]):3} Degrees Orientation (0 is x Axis)",
            _tensor_to_2Dnumpy_array(layer),
        )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


@torch.no_grad()
def save_3Dtensor_as_image_grid(
    tensor: torch.Tensor, filename: str = "Image.png"
) -> None:
    imgs = []
    for i in range(tensor.shape[0]):
        imgs.append(_normalized_array_to_grayscale(_tensor_to_2Dnumpy_array(tensor[i])))
    ncols = math.ceil(math.sqrt(len(imgs)))
    cv2.imwrite(filename, _make_grid(imgs, ncols))

    with open(f"{splitext(filename)[0]}_cols_{ncols}", "w") as file:
        file.write(str(tensor.shape))


def save_2Darray_as_image(array: np.ndarray, filename: str = "Image.png") -> None:
    cv2.imwrite(
        filename, _normalized_array_to_grayscale(_normalized_array_to_grayscale(array))
    )


@torch.no_grad()
def save_2Dtensor_as_image(tensor: torch.Tensor, filename: str = "Image.png") -> None:
    cv2.imwrite(
        filename, _normalized_array_to_grayscale(_tensor_to_2Dnumpy_array(tensor))
    )


@torch.no_grad()
def show_tensor_as_image(
    tensor: torch.Tensor, winname: str = "Image", wait=True
) -> None:
    if not INTERACTIVE_VIS:
        return
    if len(tensor.shape) == 2:
        cv2.imshow(
            winname, _normalized_array_to_grayscale(_tensor_to_2Dnumpy_array(tensor))
        )
    elif len(tensor.shape) == 3:
        imgs = []
        for i in range(tensor.shape[0]):
            imgs.append(
                _normalized_array_to_grayscale(_tensor_to_2Dnumpy_array(tensor[i]))
            )
        ncols = math.ceil(math.sqrt(len(imgs)))
        cv2.imshow(winname, _make_grid(imgs, ncols=ncols))
    if wait:
        k = cv2.waitKey(0)
        cv2.destroyWindow(winname)
        if chr(k) == "q":
            exit(0)
