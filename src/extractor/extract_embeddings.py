import torch
import tqdm

from src.utils.torch_helpers import get_dataloader_args, get_device
from src.data.embedding_dataset import BiometricEmbedding, EmbeddingDataset
from src.data.biometric_dataset import Identifier
from src.data.biometric_dataset import BiometricDataset
from src.models.DeepFinger import DeepFingerOutput


def _make_embeddings_list(new_embeddings: torch.Tensor, new_ids: list[Identifier]):
        emb_vectors = (
            new_embeddings.detach()
            .to(device=torch.device("cpu"))
            .numpy()
        )
        return [
            BiometricEmbedding(bid, vec) for bid, vec in zip(new_ids, emb_vectors)
        ]

def extract_embeddings(
    model: torch.nn.Module, eval_dataset: BiometricDataset
) -> tuple[EmbeddingDataset, EmbeddingDataset]:
    """
    Calculates similarity scores for the given dataset and outputs them to data/embeddings/<filename>

    The directory is created if it does not exits. Any existing files in the directory are deleted.
    """
    texture_embeddings = []
    minutia_embeddings = []

    model = model.to(get_device())
    dataloader = torch.utils.data.DataLoader(
        eval_dataset, **get_dataloader_args(train=False)
    )
    model.eval()  # No longer outputs logits and minutia map in eval mode
    with torch.no_grad():
        for batch_subjects, batch_impressions, vals in tqdm.tqdm(dataloader):
            fp_imgs = vals
            fp_imgs: torch.Tensor = fp_imgs.to(get_device())
            output: DeepFingerOutput = model(fp_imgs)
            ids = [
                Identifier(s, i)
                for s, i in zip(batch_subjects.tolist(), batch_impressions.tolist())
            ]

            if output.texture_embeddings is not None:
                 texture_embeddings += _make_embeddings_list(output.texture_embeddings, ids)
            if output.minutia_embeddings is not None:
                 minutia_embeddings += _make_embeddings_list(output.minutia_embeddings, ids)

    return (
        EmbeddingDataset(texture_embeddings) if len(texture_embeddings) > 0 else None,
        EmbeddingDataset(minutia_embeddings) if len(minutia_embeddings) > 0 else None,
    )
