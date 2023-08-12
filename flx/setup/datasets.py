from typing import Iterable, Union

import torch

from flx.data.transformed_dataset import TransformedDataset

from flx.data.image_helpers import (
    pad_and_resize_to_deepfinger_input_size,
)
from flx.preprocessing.binarization import (
    LazilyAllocatedBinarizer,
)
from flx.preprocessing.augmentation import RandomPoseTransform, RandomQualityTransform
from flx.data.biometric_dataset import (
    Identifier,
    IdentifierSet,
    BiometricDataset,
    ZippedDataset,
    MergedDataset,
    FilteredDataset,
    DummyDataset,
)
from flx.data.fingerprint_dataset import (
    SFingeDataset,
    FVC2004Dataset,
    MCYTCapacitiveDataset,
    MCYTOpticalDataset,
    NistSD4Dataset,
    NistSD14Dataset,
)
from flx.data.label_dataset import LabelDataset
from flx.data.minutia_map_dataset import (
    SFingeMinutiaMapDataset,
    MCYTCapacitiveMinutiaMapDataset,
    MCYTOpticalMinutiaMapDataset,
    MINUTIA_MAP_SIZE,
    MINUTIA_MAP_CHANNELS,
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
NIST_SD14_BINARIZATION = LazilyAllocatedBinarizer(1.5)


def get_training_set(
    sfinge_dir: str, mcyt_optical_dir: str, mcyt_capacitive_dir: str
) -> BiometricDataset:
    # SFinge images
    NUM_SFINGE_SUBJECTS = 6000
    sfinge_ids = _make_identifiers(range(NUM_SFINGE_SUBJECTS), range(10))
    ds_sfinge = FilteredDataset(SFingeDataset(sfinge_dir), sfinge_ids)
    ds_sfinge = TransformedDataset(
        images=ds_sfinge,
        poses=POSE_AUGMENTATION,
        transforms=[
            QUALITY_AUGMENTATION,
            SFINGE_BINARIZATION,
            pad_and_resize_to_deepfinger_input_size,
        ],
    )

    # MCYT images
    NUM_MCYT_SUBJECTS = 2000
    ds_optical = MCYTOpticalDataset(mcyt_optical_dir)
    ds_optical = FilteredDataset(
        ds_optical, ds_optical.ids.filter_by_index(range(NUM_MCYT_SUBJECTS * 12))
    )
    ds_optical = TransformedDataset(
        images=ds_optical,
        poses=POSE_AUGMENTATION,
        transforms=[QUALITY_AUGMENTATION, MCYT_OPTICAL_BINARIZATION],
    )

    ds_capacitive = MCYTCapacitiveDataset(mcyt_capacitive_dir)
    ds_capacitive = FilteredDataset(
        ds_capacitive, ds_capacitive.ids.filter_by_index(range(NUM_MCYT_SUBJECTS * 12))
    )
    ds_capacitive = TransformedDataset(
        images=ds_capacitive,
        poses=POSE_AUGMENTATION,
        transforms=[QUALITY_AUGMENTATION, MCYT_CAPACITIVE_BINARIZATION],
    )

    ds_mcyt = MergedDataset([ds_optical, ds_capacitive], share_subjects=True)
    ds = MergedDataset([ds_sfinge, ds_mcyt], share_subjects=False)

    # Minutia Maps
    ds_mm_sfinge = FilteredDataset(SFingeMinutiaMapDataset(sfinge_dir), sfinge_ids)
    assert ds_mm_sfinge.ids == ds_sfinge.ids

    ds_mm_optical = MCYTOpticalMinutiaMapDataset(mcyt_optical_dir)
    ds_mm_optical = FilteredDataset(
        ds_mm_optical, ds_mm_optical.ids.filter_by_index(range(NUM_MCYT_SUBJECTS * 12))
    )

    ds_mm_capacitive = MCYTCapacitiveMinutiaMapDataset(mcyt_capacitive_dir)
    ds_mm_capacitive = FilteredDataset(
        ds_mm_capacitive,
        ds_mm_capacitive.ids.filter_by_index(range(NUM_MCYT_SUBJECTS * 12)),
    )
    assert ds_mm_capacitive.ids == ds_capacitive.ids

    #ds_mm_mcyt = MergedDataset([ds_mm_optical, ds_mm_capacitive], share_subjects=True)
    ds_mm_mcyt = DummyDataset(ds_mcyt.ids, torch.tensor([]))
    ds_mm = MergedDataset([ds_mm_sfinge, ds_mm_mcyt], share_subjects=False)

    # Labels
    labels = LabelDataset(ds.ids)

    assert ds.num_subjects == NUM_SFINGE_SUBJECTS + NUM_MCYT_SUBJECTS
    assert len(ds) == NUM_SFINGE_SUBJECTS * 10 + NUM_MCYT_SUBJECTS * 12 * 2
    assert ds.ids == ds_mm.ids
    return (ds, ds_mm, labels)





#  ----------------- TESTING ----------------------


def _make_sfinge_no_background_testing(
    root_dir: str, subjects: Iterable[int], impressions: Iterable[int]
) -> BiometricDataset:
    ds = FilteredDataset(
        SFingeDataset(root_dir), _make_identifiers(subjects, impressions)
    )
    return TransformedDataset(
        images=ds,
        poses=None,
        transforms=[SFINGE_BINARIZATION, pad_and_resize_to_deepfinger_input_size],
    )


def get_sfinge_example(root_dir: str) -> BiometricDataset:
    return _make_sfinge_no_background_testing(
        root_dir=root_dir, subjects=range(4), impressions=range(2)
    )


def get_sfinge_validation_separate_subjects(root_dir: str) -> BiometricDataset:
    return _make_sfinge_no_background_testing(
        root_dir=root_dir, subjects=range(42000, 44000), impressions=range(4)
    )


def get_sfinge_test_none(root_dir: str) -> TransformedDataset:
    return _make_sfinge_no_background_testing(
        root_dir=root_dir, subjects=range(1000), impressions=range(4)
    )


def get_sfinge_test_optical(root_dir: str) -> TransformedDataset:
    ds = FilteredDataset(
        SFingeDataset(root_dir), _make_identifiers(range(1000), range(4))
    )
    return TransformedDataset(
        images=ds,
        poses=None,
        transforms=[SFINGE_BINARIZATION, pad_and_resize_to_deepfinger_input_size],
    )


def get_sfinge_test_capacitive(root_dir: str) -> TransformedDataset:
    ds = FilteredDataset(
        SFingeDataset(root_dir), _make_identifiers(range(1000), range(4))
    )
    return TransformedDataset(
        images=ds,
        poses=None,
        transforms=[SFINGE_BINARIZATION, pad_and_resize_to_deepfinger_input_size],
    )



def _get_mcyt(loaded_ds: BiometricDataset, poses: RandomPoseTransform = None, only_last_n: int = None) -> TransformedDataset:
    NUM_MCYT = 3300
    if only_last_n is None:
        only_last_n = NUM_MCYT
    start_index = NUM_MCYT - only_last_n

    loaded_ds = FilteredDataset(
        loaded_ds,
        loaded_ds.ids.filter_by_index(range(start_index * 12, NUM_MCYT * 12)),
    )
    assert loaded_ds.num_subjects == only_last_n
    assert len(loaded_ds) == only_last_n * 12
    return TransformedDataset(
        images=loaded_ds,
        poses=poses,
        transforms=[MCYT_OPTICAL_BINARIZATION, pad_and_resize_to_deepfinger_input_size],
    )


def get_mcyt_optical(root_dir: str, poses: RandomPoseTransform = None, only_last_n: int = None) -> TransformedDataset:
    ds = MCYTOpticalDataset(root_dir)
    return _get_mcyt(ds, poses, only_last_n)


def get_mcyt_capacitive(root_dir: str, poses: RandomPoseTransform = None, only_last_n: int = None) -> TransformedDataset:
    ds = MCYTCapacitiveDataset(root_dir)
    return _get_mcyt(ds, poses, only_last_n)


def get_fvc2004_db1a(root_dir: str) -> TransformedDataset:
    NUM_SUBJECTS = 100
    ds = FVC2004Dataset(root_dir)
    assert ds.num_subjects == NUM_SUBJECTS
    assert len(ds) == NUM_SUBJECTS * 8
    return TransformedDataset(
        images=ds,
        poses=None,
        transforms=[FVC_BINARIZATION],
    )


def get_nist_sd4(root_dir: str) -> TransformedDataset:
    NUM_SUBJECTS = 2000
    ds = NistSD4Dataset(root_dir)
    assert ds.num_subjects == NUM_SUBJECTS
    assert len(ds) == NUM_SUBJECTS * 2
    return TransformedDataset(
        images=ds,
        poses=None,
        transforms=[NIST_SD4_BINARIZATION],
    )


def get_nist_sd14(root_dir: str) -> TransformedDataset:
    NUM_SUBJECTS = 27000
    ds = NistSD14Dataset(root_dir)
    assert ds.num_subjects == NUM_SUBJECTS
    assert len(ds) == NUM_SUBJECTS * 2
    return TransformedDataset(
        images=ds,
        poses=None,
        transforms=[NIST_SD14_BINARIZATION],
    )
