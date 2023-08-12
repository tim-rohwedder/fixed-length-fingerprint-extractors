from flx.setup.experiments import TESTSETS, EXTRACTORS

from flx.setup.datasets import get_training_set
from flx.setup.paths import get_fingerprint_dataset_path

EXTRACTOR = EXTRACTORS["DeepPrint_Tex_1024"]

VALIDATION_SET = TESTSETS["SFingev2ValidationSeparateSubjects"]
NUM_EPOCHS = 75


def main():
    extractor = EXTRACTOR.load()
    output_dir = EXTRACTOR.get_dir()
    fingerprints, minutia_maps, labels = get_training_set(
        get_fingerprint_dataset_path("SFingev2"),
        get_fingerprint_dataset_path("mcyt330_optical"),
        get_fingerprint_dataset_path("mcyt330_capacitive"),
    )

    validation_set = VALIDATION_SET.load()
    validation_benchmark = VALIDATION_SET.load_verification_benchmark()
    extractor.fit(
        fingerprints=fingerprints,
        minutia_maps=minutia_maps,
        labels=labels,
        validation_fingerprints=validation_set,
        validation_benchmark=validation_benchmark,
        num_epochs=NUM_EPOCHS,
        out_dir=output_dir,
    )


if __name__ == "__main__":
    main()
