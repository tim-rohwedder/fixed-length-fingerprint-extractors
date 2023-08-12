from os.path import join

import cv2
import numpy as np
import torch

from flx.data.embedding_dataset import EmbeddingDataset, combine_embeddings
from flx.data.biometric_dataset import Identifier
from flx.data.file_dataset import FileDataset
from flx.extractor.extract_embeddings import extract_embeddings
from flx.models.DeepFinger import DeepFinger_Tex, _Branch_TextureEmbedding
from flx.visualization.show_with_opencv import (
    save_2Dtensor_as_image,
    save_3Dtensor_as_image_grid,
    save_2Darray_as_image,
)
from flx.setup.paths import created_dir
from flx.utils.torch_helpers import get_device, load_samples_to_tensor

from flx.setup.config import (
    CURRENT_VIS_MODE,
    VisualizationMode,
    make_visualization_subset_training,
)


@created_dir
def _get_layer_visualizations_dir(
    output_dir: str, epoch: int, identifier: Identifier = None
) -> str:
    base = join(output_dir, f"epoch_{epoch}")
    if identifier is None:
        return base
    return join(base, f"{identifier.subject:07}_{identifier.impression:02}")


def _combine_1Dvisualizations_over_time(filepaths: list[str], outfile: str) -> None:
    """
    Loads the 1D array visualizations and concatenates them horizontally with one red line between each array.
    """
    arrs: list[np.ndarray] = []
    for f in filepaths:
        arr = cv2.imread(f)
        arrs.append(np.reshape(arr, (-1, 1)))
        arrs.append(np.zeros_like(arrs[-1]))
    img = np.hstack(arrs)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img[:, 1::2, 2] = 255
    cv2.imsave(outfile, img)


def _visualize_layers(
    outdirs: list[str],
    input: torch.Tensor,
    layers: list[torch.nn.Module],
    layer_names: list[str],
) -> torch.Tensor:
    """
    Processes the given input tensor and for all the contained samples visualizes the outputs of all intermediate layers and saves them to disk.

    @param outdirs : Full path to the output dirs of the sample. Each sample must have its own output dir
    @param input : The input that will be passed through the layers
    @param layers : The layers in the order in which the input should pass them.
    @param layer_names : The names of the layers
    """
    assert len(outdirs) == input.shape[0]
    with torch.no_grad():
        for layer, layer_name in zip(layers, layer_names):
            input = layer(input)
            for i, sample_outdir in enumerate(outdirs):
                filename = join(sample_outdir, layer_name + ".png")
                x = input[i]
                if len(x.shape) == 1:
                    save_2Dtensor_as_image(
                        x.reshape((x.shape[0], 1)), filename=filename
                    )
                    x_arr = x.to(torch.device("cpu")).numpy()
                    np.save(filename + ".npy", x_arr)
                elif len(x.shape) == 2:
                    save_2Dtensor_as_image(x, filename=filename)
                elif len(x.shape) == 3:
                    save_3Dtensor_as_image_grid(x, filename=filename)
                else:
                    raise RuntimeError(f"Cannot visualize tensor with shape: {x.shape}")
    return input


def _get_Branch_TextureEmbedding_layers(
    module: _Branch_TextureEmbedding,
) -> tuple[list[torch.nn.Module], list[str]]:
    layers = [
        module._0_block,
        module._1_block,
        module._2_block,
        module._3_avg_pool2d,
        module._4_flatten,
        module._5_dropout,
        module._6_linear,
        lambda x: torch.nn.functional.normalize(torch.squeeze(x), dim=1),
    ]
    names = [
        "0_block",
        "1_block",
        "2_block",
        "3_avg_pool2d",
        "4_flatten",
        "5_dropout",
        "6_linear",
        "7_nomalization",
    ]
    return (layers, names)


def _visualize_center_loss(centers: torch.Tensor, outdir: str) -> None:
    save_2Dtensor_as_image(centers, join(outdir, "centers.png"))


def _visualize_embeddings(
    embeddings_texture: EmbeddingDataset,
    embeddings_minutia: EmbeddingDataset,
    outdir: str,
) -> None:
    emb = embeddings_texture
    if embeddings_minutia is not None:
        emb = combine_embeddings(emb, embeddings_minutia)
    save_2Darray_as_image(emb.numpy(), join(outdir, "embeddings.png"))


def _visualize_DeepFinger_Texture(
    model: DeepFinger_Tex, input: torch.Tensor, outdirs: list[str]
) -> None:
    """
    Visualizes the given ids which are already loaded into a tensor.
    """
    layers = []
    layers.append(torch.nn.Identity())
    layer_names = []
    layer_names.append("0_input")

    layers.append(model.stem.features)
    layer_names.append("1_stem")

    layers_texture, names_texture = _get_Branch_TextureEmbedding_layers(
        model.texture_branch
    )
    names_texture = ["2_texture." + n for n in names_texture]
    layers += layers_texture
    layer_names += names_texture

    output = _visualize_layers(
        outdirs=outdirs, input=input, layers=layers, layer_names=layer_names
    )
    return output


def visualize_training_progress(
    dataset: FileDataset,
    model: torch.nn.Module,
    model_loss: torch.nn.Module,
    epoch: int,
    visualizations_output_dir: str,
) -> None:
    if CURRENT_VIS_MODE == VisualizationMode.OFF:
        return
    model.eval()
    dataset = make_visualization_subset_training(dataset)
    outdirs = [
        _get_layer_visualizations_dir(visualizations_output_dir, epoch, id)
        for id in dataset.ids
    ]
    _, _, images = load_samples_to_tensor(dataset=dataset)

    if isinstance(model, DeepFinger_Tex):
        _visualize_center_loss(
            model_loss.texture_loss_fun.center_loss_fun.centers,
            outdir=_get_layer_visualizations_dir(visualizations_output_dir, epoch),
        )
        if CURRENT_VIS_MODE == VisualizationMode.ALL:
            output = _visualize_DeepFinger_Texture(
                model, input=images.to(get_device()), outdirs=outdirs
            )
            save_2Dtensor_as_image(
                output,
                join(
                    _get_layer_visualizations_dir(visualizations_output_dir, epoch),
                    "texture_output.png",
                ),
            )

    texture, minutia = extract_embeddings(model, dataset)
    _visualize_embeddings(
        texture,
        minutia,
        outdir=_get_layer_visualizations_dir(visualizations_output_dir, epoch),
    )
