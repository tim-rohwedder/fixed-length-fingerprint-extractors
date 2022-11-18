import json


from src.data.biometric_dataset import Identifier, BiometricDataset

from src.preprocessing.augmentation import PoseTransform


def _poses_to_json(poses: list[PoseTransform]) -> dict:
    return {
        "array_pad": [p.pad for p in poses],
        "array_angle": [p.angle for p in poses],
        "array_shift_horizontal": [p.shift_horizontal for p in poses],
        "array_shift_vertical": [p.shift_vertical for p in poses],
    }


def _poses_from_json(json: dict) -> list[PoseTransform]:
    paddings = json["array_pad"]
    angles = json["array_angle"]
    shifts_horizontal = json["array_shift_horizontal"]
    shifts_vertical = json["array_shift_vertical"]
    return [
        PoseTransform(p, r, sh, sv)
        for p, r, sh, sv in zip(paddings, angles, shifts_horizontal, shifts_vertical)
    ]


class PoseDataset(BiometricDataset):
    """
    A biometric dataset which contains fingerprint poses for given
    fingerprint samples.

    Each pose consists of a rotation and a shift in x and y direction.
    The dataset can be save to or loaded from json.
    """

    def __init__(self, ids: list[Identifier], poses: list[PoseTransform]):
        self._ids = sorted(ids)
        self._id_to_pose = {bid: pose for bid, pose in zip(ids, poses)}

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> list[Identifier]:
        return self._ids

    def get(self, identifier: Identifier) -> PoseTransform:
        return self._id_to_pose[identifier]

    def save(self, path: str) -> None:
        with open(path, "w") as file:
            obj = {
                "poses": _poses_to_json(self._id_to_pose.values()),
                "ids": Identifier.ids_to_json(self._id_to_pose.keys()),
            }
            json.dump(obj, file)

    @staticmethod
    def load(path: str) -> "PoseDataset":
        with open(path, "r") as file:
            obj = json.load(file)
            ids = Identifier.ids_from_json(obj["ids"])
            poses = _poses_from_json(obj["poses"])
            return PoseDataset(ids, poses)
