from flx.data.embedding_loader import EmbeddingLoader, Embedding
from flx.setup.experiments import (
    get_experiments,
    get_reweighting_experiments,
    ReweightingExperiment,
    Experiment,
)
from flx.reweighting.linear_regression_reweighting import (
    reweight_and_normalize_embeddings,
)


def reweight_embeddings(experiments: dict[tuple[str, str], Experiment]) -> None:
    rw_experiments: dict[
        tuple[str, str], ReweightingExperiment
    ] = get_reweighting_experiments(experiments)
    for key, exp in experiments.items():
        print(f"\nReweighting embeddings of {exp.dataset_name} with {exp.model_name}.")
        ds_train = exp.load_training_embeddings()
        ds_all = exp.load_embeddings()
        new_embeddings = reweight_and_normalize_embeddings(
            ds_train.numpy(), ds_all.numpy(), [bid.subject for bid in ds_train.ids]
        )

        new_ds = EmbeddingLoader(
            [Embedding(bid, vec) for bid, vec in zip(ds_all.ids, new_embeddings)]
        )
        rw_experiments[key].save_embeddings(new_ds)


def main() -> None:
    EXTRACTOR_KEYS = [
        "DeepPrint_Tex_256",
        "DeepPrint_Tex_512",
        "DeepPrint_NoLoc_Mixed",
    ]

    TESTSET_KEYS = [
        "mcyt330_optical",
        "mcyt330_capacitive",
    ]

    _, _, experiments = get_experiments(
        testset_keys=TESTSET_KEYS, extractor_keys=EXTRACTOR_KEYS
    )
    reweight_embeddings(experiments)


if __name__ == "__main__":
    main()
