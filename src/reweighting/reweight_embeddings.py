from src.data.embedding_dataset import EmbeddingDataset, BiometricEmbedding
from src.setup.experiments import get_experiments, get_reweighting_experiments, ReweightingExperiment, Experiment
from src.reweighting.linear_regression_reweighting import reweight_and_normalize_embeddings


def reweight_embeddings(experiments: dict[tuple[str, str], Experiment]) -> None:
    rw_experiments: dict[tuple[str, str], ReweightingExperiment] = get_reweighting_experiments(experiments)
    for key, exp in experiments.items():
        print(f"\nReweighting embeddings of {exp.dataset_name} with {exp.model_name}.")
        ds_train = exp.load_training_embeddings()
        ds_all = exp.load_embeddings()
        new_embeddings = reweight_and_normalize_embeddings(ds_train.numpy(), ds_all.numpy(), [bid.subject for bid in ds_train.ids])
        
        new_ds = EmbeddingDataset([
            BiometricEmbedding(bid, vec) for bid, vec in zip(ds_all.ids, new_embeddings)
        ])
        rw_experiments[key].save_embeddings(new_ds)



def main() -> None:
    EXTRACTOR_KEYS = [
        "DeepFinger_Tex_256",
        "DeepFinger_Tex_512",
        "DeepFinger_NoLoc_Mixed",
        ]

    TESTSET_KEYS = [
        "mcyt330_optical",
        "mcyt330_capacitive",
        "NIST SD14",
    ]

    _, _, experiments = get_experiments(
        testset_keys=TESTSET_KEYS, extractor_keys=EXTRACTOR_KEYS
    )
    reweight_embeddings(experiments)



if __name__=="__main__":
    main()