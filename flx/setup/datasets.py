from typing import Iterable

import torch

from flx.data.transformed_image_loader import TransformedImageLoader

from flx.data.image_helpers import (
    pad_and_resize_to_deepprint_input_size,
)
from flx.image_processing.binarization import (
    LazilyAllocatedBinarizer,
)
from flx.image_processing.augmentation import RandomPoseTransform, RandomQualityTransform
from flx.data.dataset import (
    Identifier,
    IdentifierSet,
    Dataset,
    ConstantDataLoader,
)
from flx.data.image_loader import (
    SFingeLoader,
    FVC2004Loader,
    MCYTCapacitiveLoader,
    MCYTOpticalLoader,
    NistSD4Dataset,
)
from flx.data.label_index import LabelIndex
from flx.data.minutia_map_loader import (
    SFingeMinutiaMapLoader,
    MCYTCapacitiveMinutiaMapLoader,
    MCYTOpticalMinutiaMapLoader,
)


def _make_identifiers(
    subjects: Iterable[int], impressions: Iterable[int]
) -> IdentifierSet:
    return IdentifierSet([Identifier(s, i) for s in subjects for i in impressions])


#  ----------------- TRAINING ----------------------

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

SFINGE_BINARIZATION = LazilyAllocatedBinarizer(5.0)
MCYT_CAPACITIVE_BINARIZATION = LazilyAllocatedBinarizer(4.8)
MCYT_OPTICAL_BINARIZATION = LazilyAllocatedBinarizer(3.8)
FVC_BINARIZATION = LazilyAllocatedBinarizer(1.8)
NIST_SD4_BINARIZATION = LazilyAllocatedBinarizer(4.0)


def get_training_set(
    sfinge_dir: str, mcyt_optical_dir: str, mcyt_capacitive_dir: str
) -> Dataset:
    # SFinge images
    NUM_SFINGE_SUBJECTS = 6000
    sfinge_ids = _make_identifiers(range(NUM_SFINGE_SUBJECTS), range(10))
    sfinge_images = TransformedImageLoader(
        images=SFingeLoader(sfinge_dir),
        poses=POSE_AUGMENTATION,
        transforms=[
            QUALITY_AUGMENTATION,
            SFINGE_BINARIZATION,
            pad_and_resize_to_deepprint_input_size,
        ],
    )
    sfinge_images = Dataset(sfinge_images, sfinge_ids)

    # MCYT images
    NUM_MCYT_SUBJECTS = 2000
    mcyt_optical_images = TransformedImageLoader(
        images=MCYTOpticalLoader(mcyt_optical_dir),
        poses=POSE_AUGMENTATION,
        transforms=[QUALITY_AUGMENTATION, MCYT_OPTICAL_BINARIZATION],
    )
    mcyt_optical_images = Dataset(
        mcyt_optical_images,
        mcyt_optical_images.ids.filter_by_index(range(NUM_MCYT_SUBJECTS * 12)),
    )

    mcyt_capacitive_images = TransformedImageLoader(
        images=MCYTCapacitiveLoader(mcyt_capacitive_dir),
        poses=POSE_AUGMENTATION,
        transforms=[QUALITY_AUGMENTATION, MCYT_CAPACITIVE_BINARIZATION],
    )
    mcyt_capacitive_images = Dataset(
        mcyt_capacitive_images,
        mcyt_capacitive_images.ids.filter_by_index(range(NUM_MCYT_SUBJECTS * 12)),
    )

    mcyt_images = Dataset.concatenate(
        mcyt_optical_images, mcyt_capacitive_images, share_subjects=True
    )
    ds = Dataset.concatenate(sfinge_images, mcyt_images, share_subjects=False)

    # Minutia Maps
    sfinge_minumaps = Dataset(SFingeMinutiaMapLoader(sfinge_dir), sfinge_ids)

    # mcyt_optical_minumaps = MCYTOpticalMinutiaMapLoader(mcyt_optical_dir)
    # mcyt_optical_minumaps = Dataset(
    #     mcyt_optical_minumaps, mcyt_optical_minumaps.ids.filter_by_index(range(NUM_MCYT_SUBJECTS * 12))
    # )
    #
    # mcyt_capacitive_minumaps = MCYTCapacitiveMinutiaMapLoader(mcyt_capacitive_dir)
    # mcyt_capacitive_minumaps = Dataset(
    #     mcyt_capacitive_minumaps,
    #     mcyt_capacitive_minumaps.ids.filter_by_index(range(NUM_MCYT_SUBJECTS * 12)),
    # )
    # assert mcyt_capacitive_minumaps.ids == mcyt_capacitive_images.ids
    #
    # ds_mm_mcyt = MergedDataset([ds_mm_optical, ds_mm_capacitive], share_subjects=True)

    mcyt_minumaps = Dataset(ConstantDataLoader(torch.tensor([])), mcyt_images.ids)
    minumaps = Dataset.concatenate(
        sfinge_minumaps, mcyt_minumaps, share_subjects=False
    )

    # Labels
    labels = Dataset(LabelIndex(ds.ids), ds.ids)

    assert ds.num_subjects == NUM_SFINGE_SUBJECTS + NUM_MCYT_SUBJECTS
    assert len(ds) == NUM_SFINGE_SUBJECTS * 10 + NUM_MCYT_SUBJECTS * 12 * 2
    assert ds.ids == minumaps.ids
    return (ds, minumaps, labels)


#  ----------------- TESTING ----------------------


def _make_sfinge_no_background_testing(
    root_dir: str, subjects: Iterable[int], impressions: Iterable[int]
) -> Dataset:
    loader = TransformedImageLoader(
        images=SFingeLoader(root_dir),
        poses=None,
        transforms=[SFINGE_BINARIZATION, pad_and_resize_to_deepprint_input_size],
    )
    return Dataset(loader, _make_identifiers(subjects, impressions))


def get_sfinge_example(root_dir: str) -> Dataset:
    return _make_sfinge_no_background_testing(
        root_dir=root_dir, subjects=range(4), impressions=range(2)
    )


def get_sfinge_validation_separate_subjects(root_dir: str) -> Dataset:
    return _make_sfinge_no_background_testing(
        root_dir=root_dir, subjects=range(42000, 44000), impressions=range(4)
    )


def get_sfinge_test(root_dir: str) -> TransformedImageLoader:
    return _make_sfinge_no_background_testing(
        root_dir=root_dir, subjects=range(1000), impressions=range(4)
    )


def _get_mcyt(
    loader: Dataset,
    poses: RandomPoseTransform = None,
    only_last_n: int = None,
) -> TransformedImageLoader:
    loader = TransformedImageLoader(
        images=loader,
        poses=poses,
        transforms=[MCYT_OPTICAL_BINARIZATION, pad_and_resize_to_deepprint_input_size],
    )

    NUM_MCYT = 3300
    if only_last_n is None:
        only_last_n = NUM_MCYT
    start_index = NUM_MCYT - only_last_n
    dataset = Dataset(
        loader,
        loader.ids.filter_by_index(range(start_index * 12, NUM_MCYT * 12)),
    )
    assert dataset.num_subjects == only_last_n
    assert len(dataset) == only_last_n * 12
    return dataset


def get_mcyt_optical(
    root_dir: str, poses: RandomPoseTransform = None, only_last_n: int = None
) -> TransformedImageLoader:
    ds = MCYTOpticalLoader(root_dir)
    return _get_mcyt(ds, poses, only_last_n)


def get_mcyt_capacitive(
    root_dir: str, poses: RandomPoseTransform = None, only_last_n: int = None
) -> TransformedImageLoader:
    ds = MCYTCapacitiveLoader(root_dir)
    return _get_mcyt(ds, poses, only_last_n)


def get_fvc2004_db1a(root_dir: str) -> TransformedImageLoader:
    NUM_SUBJECTS = 100
    loader = TransformedImageLoader(
        images=FVC2004Loader(root_dir),
        poses=None,
        transforms=[FVC_BINARIZATION],
    )
    assert loader.ids.num_subjects == NUM_SUBJECTS
    assert len(loader.ids) == NUM_SUBJECTS * 8
    return Dataset(loader, loader.ids)


def get_nist_sd4(root_dir: str) -> TransformedImageLoader:
    NUM_SUBJECTS = 2000
    loader = TransformedImageLoader(
        images=NistSD4Dataset(root_dir),
        poses=None,
        transforms=[NIST_SD4_BINARIZATION],
    )
    assert loader.ids.num_subjects == NUM_SUBJECTS
    assert len(loader.ids) == NUM_SUBJECTS * 2
    return Dataset(loader, loader.ids)
