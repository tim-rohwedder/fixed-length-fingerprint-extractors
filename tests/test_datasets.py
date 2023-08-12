from os.path import join, dirname
from os import remove

import torch
import cv2
import numpy as np

from flx.data.transformed_dataset import TransformedImageLoader
from flx.data.dataset import Identifier, IdentifierSet
from flx.data.label_index import LabelIndex
from flx.data.embedding_loader import (
    EmbeddingLoader,
)
from flx.data.image_loader import (
    ImageLoader,
    SFingeLoader,
    FVC2004Loader,
    NistSD4Dataset,
    MCYTOpticalLoader,
    MCYTCapacitiveLoader,
)
from flx.data.minutia_map_loader import (
    SFingeMinutiaMapLoader,
    MCYTOpticalMinutiaMapLoader,
    MCYTCapacitiveMinutiaMapLoader,
)
from flx.data.pose_dataset import PoseLoader
from flx.setup.paths import get_fingerprint_dataset_path, get_pose_dataset_path
from flx.image_processing.augmentation import (
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
    label_index: LabelIndex = LabelIndex(
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
    assert len(label_index.ids) == 5
    assert label_index.ids.num_subjects == 4
    assert label_index.get(Identifier(1, 2)) == 0
    assert label_index.get(Identifier(3, 0)) == 1
    assert label_index.get(Identifier(3, 1)) == 1
    assert label_index.get(Identifier(4, 1)) == 2
    assert label_index.get(Identifier(5, 2)) == 3


def test_embedding_dataset():
    ids = [Identifier(0, 0), Identifier(0, 1), Identifier(1, 0), Identifier(1, 1)]
    embeddings = np.array([np.random.random(4) for _ in ids])

    loader = EmbeddingLoader(IdentifierSet(ids), embeddings)
    for id, emb in zip(loader.ids, loader.numpy()):
        print(id)
        print(emb)

    dataset2 = EmbeddingLoader.combine(loader, loader)
    for id, emb in zip(dataset2.ids, dataset2.numpy()):
        print(id)
        print(emb)


def test_minutia_map_sfinge():
    loader = SFingeMinutiaMapLoader(join(TEST_DATA_DIR, "SFingeExample"))
    minutia_map, mask = loader.get(Identifier(0, 0))
    print(minutia_map.shape)
    show_minutia_maps_from_tensor(minutia_map)


def test_minutia_map_mcyt_capactive():
    loader = MCYTCapacitiveMinutiaMapLoader(
        join(TEST_DATA_DIR, "MCYTCapacitiveExample")
    )
    minutia_map, mask = loader.get(Identifier(0, 0))
    print(minutia_map.shape)
    show_minutia_maps_from_tensor(minutia_map)


def test_minutia_map_mcyt_optical():
    loader = MCYTOpticalMinutiaMapLoader(join(TEST_DATA_DIR, "MCYTOpticalExample"))
    minutia_map, mask = loader.get(Identifier(0, 8))
    print(minutia_map.shape)
    show_minutia_maps_from_tensor(minutia_map)


def test_pose_dataset():
    def make_random_pose_dataset(
        ids: list[Identifier], pose_distribution: RandomPoseTransform
    ) -> PoseLoader:
        return PoseLoader(ids, [pose_distribution.sample() for _ in ids])

    print("Testing PoseDataset")
    image_loader = SFingeLoader(join(TEST_DATA_DIR, "SFingeExample"))
    pose_loader = make_random_pose_dataset(
        image_loader.ids, pose_distribution=RandomPoseTransform()
    )
    pose_loader.save(get_pose_dataset_path("PoseTest"))

    pose_dataloader = PoseLoader.load(get_pose_dataset_path("PoseTest"))
    for bid in image_loader.ids:
        fp = image_loader.get(bid)
        show_tensor_as_image(
            pose_loader.get(bid)(fp), winname=f"Random pose: {bid}", wait=False
        )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    remove(get_pose_dataset_path("PoseTest"))


def test_transformed_dataset():
    images = SFingeLoader(join(TEST_DATA_DIR, "SFingeExample"))

    transformed_images = TransformedImageLoader(images)
    for bid in transformed_images.ids:
        show_tensor_as_image(
            transformed_images.get(bid),
            winname=f"NoPoseTrafo, NoQualityTrafo: {bid}",
            wait=False,
        )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    transformed_images = TransformedImageLoader(
        images, poses=RandomPoseTransform(), transforms=[RandomQualityTransform()]
    )
    for bid in transformed_images.ids:
        show_tensor_as_image(
            transformed_images.get(bid),
            winname=f"PoseDistribution, QualityDistribution: {bid}",
            wait=False,
        )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pose_loader = PoseLoader(
        images.ids,
        [
            PoseTransform(pad=80, angle=i / len(images.ids) * 90)
            for i in range(len(images.ids))
        ],
    )
    transformed_images = TransformedImageLoader(images, poses=pose_loader)
    for bid in transformed_images.ids:
        show_tensor_as_image(
            transformed_images.get(bid),
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
    dataset: ImageLoader = dataset_type(get_fingerprint_dataset_path(root_dir_name))
    assert dataset.ids.num_subjects == expected_num_subjects
    assert len(dataset.ids) == expected_num_samples
    image: torch.Tensor = dataset.get(dataset.ids[0])
    assert_is_image_tensor(image)
    show_tensor_as_image(image, f"{root_dir_name}", wait=True)


def test_dataset_SFingeExample():
    _test_image_dataset(SFingeLoader, "test_data_dir:SFingeExample", 4, 4 * 2)


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
