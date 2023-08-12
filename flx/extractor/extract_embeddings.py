from typing import Union

import numpy as np
import torch
import tqdm

from flx.models.torch_helpers import get_dataloader_args, get_device
from flx.data.embedding_loader import EmbeddingLoader
from flx.data.dataset import IdentifierSet
from flx.data.dataset import Dataset
from flx.models.deep_print_arch import DeepPrintOutput


def _to_numpy(embeddings: torch.Tensor):
    if embeddings is None:
        return None
    return embeddings.detach().to(device=torch.device("cpu")).numpy()


def _concatenate_embeddings_if_exist(
    ids: IdentifierSet, embeddings: list[Union[None, np.ndarray]]
) -> torch.Tensor:
    embeddings_filtered = [e for e in embeddings if e is not None]
    if len(embeddings_filtered) == 0:
        return None
    if len(embeddings_filtered) != len(embeddings):
        raise ValueError("Some embeddings are None, others are not!")
    return EmbeddingLoader(ids, np.concatenate(embeddings, axis=0))


def extract_embeddings(
    model: torch.nn.Module, fingerprint_dataset: Dataset
) -> tuple[EmbeddingLoader, EmbeddingLoader]:
    texture_embeddings = []
    minutia_embeddings = []

    model = model.to(get_device())
    dataloader = torch.utils.data.DataLoader(
        fingerprint_dataset, **get_dataloader_args(train=False)
    )
    model.eval()  # No longer outputs logits and minutia map in eval mode
    with torch.no_grad():
        for vals in tqdm.tqdm(dataloader):
            fp_imgs = vals
            fp_imgs: torch.Tensor = fp_imgs.to(get_device())
            output: DeepPrintOutput = model(fp_imgs)

            texture_embeddings.append(_to_numpy(output.texture_embeddings))
            minutia_embeddings.append(_to_numpy(output.minutia_embeddings))

    return (
        _concatenate_embeddings_if_exist(fingerprint_dataset.ids, texture_embeddings),
        _concatenate_embeddings_if_exist(fingerprint_dataset.ids, minutia_embeddings),
    )
