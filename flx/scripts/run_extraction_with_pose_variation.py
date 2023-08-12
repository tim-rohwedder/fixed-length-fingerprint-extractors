from flx.generate_embeddings import generate_embeddings
from flx.image_processing.augmentation import RandomPoseTransform
from flx.setup.experiments import (
    get_experiments,
    DatasetLoader,
    FixedLengthExtractorLoader,
    Experiment,
)
from copy import deepcopy
from flx.scripts.run_reweighting import reweight_embeddings


def _get_rotation_transforms(
    angle_deviation: int, shift_deviation: int
) -> list[RandomPoseTransform]:
    return RandomPoseTransform(
        angle_min=-angle_deviation,
        angle_max=angle_deviation,
        shift_horizontal_min=-shift_deviation,
        shift_horizontal_max=shift_deviation,
        shift_vertical_min=-shift_deviation,
        shift_vertical_max=shift_deviation,
    )


def get_alignment_dataset_name(
    dataset_name: str, angle_deviation: int, shift_deviation: int
) -> str:
    return f"{dataset_name}_rot_{angle_deviation}_shift_{shift_deviation}"


def main():
    ROTATION_MAGNITUDES = [0, 15, 30, 60, 120]
    SHIFT_MAGNITUDES = [0, 10, 20, 40, 80]

    testsets, extractors, experiments = get_experiments(
        testset_keys=[
            "mcyt330_optical",
            "mcyt330_capacitive",
        ],
        extractor_keys=["DeepPrint_TexMinu_512"],
    )

    new_experiments = {}

    for dataloader in testsets:
        for extractor in extractors:
            for angle_deviation in ROTATION_MAGNITUDES:
                for shift_deviation in SHIFT_MAGNITUDES:
                    print(
                        f"\nGenerating embeddings of {dataloader.name} with {extractor.name}."
                    )
                    # Adjust the dataset loader to include the given level of rotation and translation
                    dataloader_mod: DatasetLoader = deepcopy(dataloader)
                    dataloader_mod.kwargs = {
                        "poses": _get_rotation_transforms(
                            angle_deviation, shift_deviation
                        ),
                        "only_last_n": 1300,
                    }

                    # Create new experiment with modified dataset name
                    experiment_mod: Experiment = experiments[
                        (extractor.name, dataloader.name)
                    ]
                    experiment_mod.dataset_name = get_alignment_dataset_name(
                        dataloader.name, angle_deviation, shift_deviation
                    )
                    new_experiments[
                        (extractor.name, experiment_mod.dataset_name)
                    ] = experiment_mod

                    generate_embeddings(extractor, dataloader_mod, experiment_mod)

    reweight_embeddings(new_experiments)


if __name__ == "__main__":
    main()
