from enum import Enum


from flx.data.biometric_dataset import Identifier, FilteredDataset, BiometricDataset


# VISUAL INVESTIGATION

INTERACTIVE_VIS = False  

LEARNING_RATE = 0.025


def make_visualization_subset_training(
    training_dataset: BiometricDataset,
) -> BiometricDataset:
    return FilteredDataset(
        training_dataset, training_dataset.ids.filter_by_index(range(20))
    )

# Changes this when switching models DeepFinger has 299 while ViT has 224
INPUT_SIZE = 299