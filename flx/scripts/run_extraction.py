from flx.setup.experiments import (
    get_experiments,
    DatasetLoader,
    FixedLengthExtractorLoader,
    Experiment,
)


def generate_embeddings(
    extractor_loader: FixedLengthExtractorLoader,
    dataset_loader: DatasetLoader,
    experiment: Experiment,
):
    extractor = extractor_loader.load()
    texture_embeddings, minutia_embeddings = extractor.extract(dataset_loader.load())
    experiment.save_embeddings(texture_embeddings, minutia_embeddings)


def main() -> None:
    EXTRACTOR_KEYS = ["DeepPrint_Tex_256"]

    TESTSET_KEYS = [
        "mcyt330_optical",
        "mcyt330_capacitive",
    ]

    testsets, extractors, experiments = get_experiments(
        testset_keys=TESTSET_KEYS, extractor_keys=EXTRACTOR_KEYS
    )

    for extractor in extractors:
        for dataset in testsets:
            print(f"\nGenerating embeddings of {dataset.name} with {extractor.name}.")
            generate_embeddings(
                extractor, dataset, experiments[(extractor.name, dataset.name)]
            )


if __name__ == "__main__":
    main()
