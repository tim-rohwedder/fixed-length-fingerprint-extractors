from os.path import join, dirname
from os import remove

import torch
import cv2
import numpy as np

from flx.data.transformed_dataset import TransformedDataset
from flx.data.biometric_dataset import Identifier, IdentifierSet
from flx.data.label_dataset import LabelDataset
from flx.data.embedding_dataset import (
    EmbeddingDataset,
    BiometricEmbedding,
    combine_embeddings,
)
from flx.data.fingerprint_dataset import (
    FingerprintDataset,
    SFingeDataset,
    FVC2004Dataset,
    NistSD4Dataset,
    NistSD14Dataset,
    MCYTOpticalDataset,
    MCYTCapacitiveDataset,
)
from flx.data.minutia_map_dataset import (
    SFingeMinutiaMapDataset,
    MCYTOpticalMinutiaMapDataset,
    MCYTCapacitiveMinutiaMapDataset,
)
from flx.data.pose_dataset import PoseDataset
from flx.setup.paths import get_fingerprint_dataset_path, get_pose_dataset_path
from flx.preprocessing.augmentation import (
    RandomPoseTransform,
    PoseTransform,
    RandomQualityTransform,
)
from flx.visualization.show_with_opencv import (
    show_tensor_as_image,
    show_minutia_maps_from_tensor,
)

TEST_DATA_DIR = join(dirname(__file__), "data")


def assert_is_image_tensor(tensor: torch.Tensor) -> None:
    assert tensor.ndim == 3
    assert tensor.shape[0] == 1 or tensor.shape[0] == 0
    assert tensor.shape[1] > 0
    assert tensor.shape[2] > 0


def assert_identifiers(ds: IdentifierSet, num_subjects: int, num_samples: int):
    assert len(ds) == num_samples
    assert ds.num_subjects == num_subjects


def test_identifier_set():
    all_ids = [
        Identifier(4, 1),
        Identifier(3, 1),
        Identifier(5, 2),
        Identifier(3, 0),
        Identifier(1, 2),
        Identifier(4, 0),
        Identifier(1, 0),
        Identifier(1, 1),
        Identifier(2, 0),
        Identifier(2, 1),
        Identifier(5, 0),
        Identifier(5, 1),
    ]

    ds = IdentifierSet(all_ids)
    assert_identifiers(ds, 5, 12)

    # Test filter by subject
    subset = ds.filter_by_subject([2, 4])
    assert_identifiers(subset, 2, 4)
    assert subset[0] == Identifier(2, 0)
    assert subset[1] == Identifier(2, 1)
    assert subset[2] == Identifier(4, 0)
    assert subset[3] == Identifier(4, 1)

    print("\nFilter by id")
    print("Ids for subset " + " ".join(str(i) for i in all_ids[4:6]))
    subset = ds.filter_by_id(IdentifierSet(all_ids[4:6]))
    assert_identifiers(subset, 2, 2)
    assert subset[0] == all_ids[4]
    assert subset[1] == all_ids[5]

    subset = ds.filter_by_index(range(3))
    assert_identifiers(subset, 1, 3)
    assert subset[0] == Identifier(1, 0)
    assert subset[1] == Identifier(1, 1)
    assert subset[2] == Identifier(1, 2)


def test_label_dataset():
    ds1: LabelDataset = LabelDataset(
        IdentifierSet(
            [
                Identifier(4, 1),
                Identifier(3, 1),
                Identifier(5, 2),
                Identifier(3, 0),
                Identifier(1, 2),
            ]
        )
    )
    assert len(ds1) == 5
    assert ds1.num_subjects == 4
    assert ds1.get(Identifier(1, 2)) == 0
    assert ds1.get(Identifier(3, 0)) == 1
    assert ds1.get(Identifier(3, 1)) == 1
    assert ds1.get(Identifier(4, 1)) == 2
    assert ds1.get(Identifier(5, 2)) == 3


def test_embedding_dataset():
    ids = [Identifier(0, 0), Identifier(0, 1), Identifier(1, 0), Identifier(1, 1)]
    embeddings = np.array([np.random.random(4) for _ in ids])

    dataset = EmbeddingDataset(
        [BiometricEmbedding(bid, emb) for bid, emb in zip(ids, embeddings)]
    )
    for emb in dataset.embeddings.values():
        print(emb.id)
        print(emb.vector)

    dataset2 = combine_embeddings(dataset, dataset)
    for emb in dataset2.embeddings.values():
        print(emb.id)
        print(emb.vector)


def test_minutia_map_sfinge():
    dataset = SFingeMinutiaMapDataset(join(TEST_DATA_DIR, "SFingeExample"))
    minutia_map, mask = dataset.get(Identifier(0, 0))
    print(minutia_map.shape)
    show_minutia_maps_from_tensor(minutia_map)


def test_minutia_map_mcyt_capactive():
    dataset = MCYTCapacitiveMinutiaMapDataset(
        join(TEST_DATA_DIR, "MCYTCapacitiveExample")
    )
    minutia_map, mask = dataset[0][2]
    print(minutia_map.shape)
    show_minutia_maps_from_tensor(minutia_map)


def test_minutia_map_mcyt_optical():
    dataset = MCYTOpticalMinutiaMapDataset(join(TEST_DATA_DIR, "MCYTOpticalExample"))
    minutia_map, mask = dataset[0][2]
    print(minutia_map.shape)
    show_minutia_maps_from_tensor(minutia_map)


def test_pose_dataset():
    def make_random_pose_dataset(
        ids: list[Identifier], pose_distribution: RandomPoseTransform
    ) -> PoseDataset:
        return PoseDataset(ids, [pose_distribution.sample() for _ in ids])

    print("Testing PoseDataset")
    fingerprint_dataset = SFingeDataset(join(TEST_DATA_DIR, "SFingeExample"))
    pose_dataset = make_random_pose_dataset(
        fingerprint_dataset.ids, pose_distribution=RandomPoseTransform()
    )
    pose_dataset.save(get_pose_dataset_path("PoseTest"))

    pose_dataset = PoseDataset.load(get_pose_dataset_path("PoseTest"))
    for bid in fingerprint_dataset.ids:
        fp = fingerprint_dataset.get(bid)
        show_tensor_as_image(
            pose_dataset.get(bid)(fp), winname=f"Random pose: {bid}", wait=False
        )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    remove(get_pose_dataset_path("PoseTest"))


def test_transformed_dataset():
    dataset = SFingeDataset(join(TEST_DATA_DIR, "SFingeExample"))

    transformed = TransformedDataset(dataset)
    for bid in transformed.ids:
        show_tensor_as_image(
            transformed.get(bid),
            winname=f"NoPoseTrafo, NoQualityTrafo: {bid}",
            wait=False,
        )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    transformed = TransformedDataset(
        dataset, poses=RandomPoseTransform(), transforms=[RandomQualityTransform()]
    )
    for bid in transformed.ids:
        show_tensor_as_image(
            transformed.get(bid),
            winname=f"PoseDistribution, QualityDistribution: {bid}",
            wait=False,
        )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pose_dataset = PoseDataset(
        dataset.ids,
        [
            PoseTransform(pad=80, angle=i / len(dataset) * 90)
            for i in range(len(dataset))
        ],
    )
    transformed = TransformedDataset(dataset, poses=pose_dataset)
    for bid in transformed.ids:
        show_tensor_as_image(
            transformed.get(bid),
            winname=f"PoseDataset, NoQualityTrafo: {bid}",
            wait=False,
        )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _test_image_dataset(
    dataset_type: type,
    root_dir_name: str,
    expected_num_subjects: int,
    expected_num_samples: int,
):
    if root_dir_name.startswith("test_data_dir:"):
        root_dir_name = join(TEST_DATA_DIR, root_dir_name[14:])
    dataset: FingerprintDataset = dataset_type(
        get_fingerprint_dataset_path(root_dir_name)
    )
    assert dataset.ids.num_subjects == expected_num_subjects
    assert len(dataset.ids) == expected_num_samples
    image: torch.Tensor = dataset.get(dataset.ids[0])
    assert_is_image_tensor(image)
    show_tensor_as_image(image, f"{root_dir_name}", wait=True)


def test_dataset_SFingeExample():
    _test_image_dataset(SFingeDataset, "test_data_dir:SFingeExample", 4, 4 * 2)


# # Uncomment to test that the datasets are present on the system
#
# def test_dataset_SFingev2ValidationSeparateSubjects():
#     _test_image_dataset(SFingeDataset, "SFingev2ValidationSeparateSubjects", 2000, 2000 * 4)
#
# def test_dataset_SFingev2():
#     _test_image_dataset(SFingeDataset, "SFingev2", 8000, 8000 * 10)
#
# def test_dataset_MCYT330_Optical():
#     _test_image_dataset(MCYTOpticalDataset, "mcyt330_optical", 3300, 3300 * 12)
#
# def test_dataset_MCYT330_Capacitive():
#     _test_image_dataset(MCYTCapacitiveDataset, "mcyt330_capacitive", 3300, 3300 * 12)
