from flx.setup.experiments import TESTSETS
from flx.setup.paths import get_debug_file
from flx.visualization.show_with_opencv import save_2Dtensor_as_image
from flx.preprocessing.augmentation import RandomPoseTransform, RandomQualityTransform
from flx.setup.datasets import SFINGE_BINARIZATION
from flx.data.image_helpers import pad_and_resize_to_deepfinger_input_size
from flx.data.transformed_dataset import TransformedDataset
from flx.data.fingerprint_dataset import SFingeDataset
from flx.setup.paths import get_fingerprint_dataset_path


TESTSET_KEYS = [
    "FVC2004_DB1A",
    "mcyt330_optical",
    "mcyt330_capacitive",
    "SFingev2TestNone",
    "SFingev2TestCapacitive",
    "SFingev2TestOptical",
    "SFingev2ValidationSeparateSubjects",
    "NIST SD4",
    "NIST SD14",
]

TESTSET_KEYS = [
    "NIST SD4",
]


def show_preprocessed_samples():
    for testset_key in TESTSET_KEYS:
        ds = TESTSETS[testset_key].load()
        for i in range(8):
            subject, impression, img = ds[i]
            save_2Dtensor_as_image(
                img, get_debug_file(testset_key, f"{subject}_{impression}.png")
            )


def show_training_samples():
    QUALITY_AUGMENTATION = RandomQualityTransform(
        contrast_min=1.3, contrast_max=2.0, gain_min=0.95, gain_max=1.05
    )

    POSE_AUGMENTATION = RandomPoseTransform(
        pad=0,
        angle_min=-15,
        angle_max=15,
        shift_horizontal_min=-25,
        shift_horizontal_max=25,
        shift_vertical_min=-25,
        shift_vertical_max=25,
    )

    TRAINING_SET = TransformedDataset(
        images=SFingeDataset(get_fingerprint_dataset_path("SFingev2")),
        poses=POSE_AUGMENTATION,
        transforms=[
            QUALITY_AUGMENTATION,
            SFINGE_BINARIZATION,
            pad_and_resize_to_deepfinger_input_size,
        ],
    )

    for i in range(10):
        for j in range(4):
            img, _, _, subject, impression = TRAINING_SET[i]
            save_2Dtensor_as_image(
                img, get_debug_file("training_set", f"{subject}_{impression}_{j}.png")
            )


def main():
    show_preprocessed_samples()
    # show_training_samples()


if __name__ == "__main__":
    main()
