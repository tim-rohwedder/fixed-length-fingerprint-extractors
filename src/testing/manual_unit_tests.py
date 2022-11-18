from os.path import join, dirname
from os import remove

import cv2
import numpy as np

from src.data.transformed_dataset import TransformedDataset
from src.data.biometric_dataset import Identifier, IdentifierSet
from src.data.label_dataset import LabelDataset
from src.data.embedding_dataset import (
    EmbeddingDataset,
    BiometricEmbedding,
    combine_embeddings,
)
from src.data.fingerprint_dataset import (
    SFingeDataset,
    FVC2004Dataset,
    NistSD4Dataset,
    NistSD14Dataset,
    MCYTOpticalDataset,
    MCYTCapacitiveDataset,
)
from src.data.minutia_map_dataset import (
    SFingeMinutiaMapDataset,
    MCYTOpticalMinutiaMapDataset,
    MCYTCapacitiveMinutiaMapDataset,
)
from src.data.pose_dataset import PoseDataset
from src.setup.paths import get_fingerprint_dataset_path, get_pose_dataset_path
from src.preprocessing.augmentation import (
    RandomPoseTransform,
    PoseTransform,
    RandomQualityTransform,
)
from src.visualization.show_with_opencv import (
    show_tensor_as_image,
    show_minutia_maps_from_tensor,
    save_3Dtensor_as_image_grid,
)

TEST_DATA_DIR = join(dirname(__file__), "data")


def test_identifier_set():
    def print_set(ds: IdentifierSet):
        print(f"number of samples: {len(ds)}")
        print(f"number of subjects: {ds.num_subjects}")
        for idx in range(len(ds)):
            print(f"index={idx} id={ds[idx]}")

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

    print("\nIdentifier set")
    ds = IdentifierSet(all_ids)
    print_set(ds)

    print("\nFilter by subject")
    subjects = [1, 3, 4, 5]
    print(f"Subjects for subset: {subjects}")
    subset = ds.filter_by_subject([2, 4])
    print_set(subset)

    print("\nFilter by id")
    print("Ids for subset " + " ".join(str(i) for i in all_ids[4:6]))
    subset = ds.filter_by_id(IdentifierSet(all_ids[4:6]))
    print_set(subset)

    print("\nFilter by index")
    print("Indices for subset " + " ".join(str(i) for i in range(3)))
    subset = ds.filter_by_index(range(3))
    print_set(subset)


def test_label_dataset():
    def print_dataset(ds: LabelDataset):
        print(f"number of samples: {len(ds)}")
        print(f"number of subjects: {ds.num_subjects}")
        for idx in range(len(ds)):
            print(f"index={idx} id={ds.ids[idx]} label={ds.get(ds.ids[idx])}")

    ds1 = LabelDataset(
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
    ds2 = LabelDataset(
        IdentifierSet(
            [
                Identifier(4, 0),
                Identifier(1, 0),
                Identifier(1, 1),
                Identifier(2, 0),
                Identifier(2, 1),
                Identifier(5, 0),
                Identifier(5, 1),
            ]
        )
    )

    print("\nDataset 1")
    print_dataset(ds1)
    print("\nDataset 2")
    print_dataset(ds2)


def test_embedding_dataset():
    print("Testing EmbeddingDataset")
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


def test_image_dataset(dataset_type: type, root_dir_name: str):
    print(f"Testing {dataset_type}({root_dir_name})")
    dataset = dataset_type(get_fingerprint_dataset_path(root_dir_name))
    if len(dataset) == 0:
        return
    image = dataset.get(dataset.ids[0])
    print(image.shape)
    show_tensor_as_image(image, f"{root_dir_name}", wait=False)


def test_minutia_map_dataset():
    print("Testing ISTMinutiaMapDataset")
    dataset = SFingeMinutiaMapDataset(join(TEST_DATA_DIR, "SFingev2Example"))
    minutia_map, mask = dataset.get(Identifier(0, 0))
    print(minutia_map.shape)
    show_minutia_maps_from_tensor(minutia_map)

    print("Testing MCYTMinutiaMapDataset")
    dataset = MCYTCapacitiveMinutiaMapDataset(
        join(TEST_DATA_DIR, "MCYTCapacitiveExample")
    )
    minutia_map, mask = dataset[0][2]
    print(minutia_map.shape)
    show_minutia_maps_from_tensor(minutia_map)

    print("Testing MCYTMinutiaMapDataset")
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
    fingerprint_dataset = SFingeDataset(join(TEST_DATA_DIR, "SFingev2Example"))
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
    dataset = SFingeDataset(join(TEST_DATA_DIR, "SFingev2Example"))

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


def main():
    test_minutia_map_dataset()
    exit(0)
    test_identifier_set()
    test_label_dataset()
    test_embedding_dataset()
    test_image_dataset(SFingeDataset, "SFingev2Example")
    test_image_dataset(SFingeDataset, "SFingev2")
    test_image_dataset(FVC2004Dataset, "FVC2004_DB1A")
    test_image_dataset(NistSD4Dataset, "NIST SD4")
    test_image_dataset(NistSD14Dataset, "NIST SD14")
    test_image_dataset(MCYTOpticalDataset, "mcyt330_optical")
    test_image_dataset(MCYTCapacitiveDataset, "mcyt330_capacitive")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    test_transformed_dataset()
    test_pose_dataset()


if __name__ == "__main__":
    main()
