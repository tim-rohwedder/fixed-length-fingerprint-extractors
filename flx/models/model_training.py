from dataclasses import dataclass
import json
from os.path import join, exists

import tqdm
import shutil

import torch
import torchmetrics

from flx.setup.paths import get_best_model_file
from flx.setup.paths import get_newest_model_file
from flx.benchmarks.matchers import CosineSimilarityMatcher
from flx.benchmarks.verification import VerificationBenchmark, VerificationResult
from flx.data.embedding_loader import EmbeddingLoader
from flx.data.dataset import Dataset, ZippedDataLoader
from flx.extractor.extract_embeddings import extract_embeddings
from flx.models.deep_print_arch import DeepPrintTrainingOutput
from flx.setup.config import LEARNING_RATE
from flx.models.torch_helpers import (
    get_device,
    load_model_parameters,
    save_model_parameters,
    get_dataloader_args,
)


@dataclass
class TrainingLogEntry:
    epoch: int
    training_loss: float
    loss_statistics: float
    training_accuracy: float
    validation_equal_error_rate: float

    def __str__(self):
        s = "TrainingLogEntry(\n"
        for k, v in self.__dict__.items():
            s += f"    {k}={v},\n"
        return s + "}"


class TrainingLog:
    def __init__(self, path: str, reset: bool = False):
        self._path: str = path
        self._entries: list[TrainingLogEntry] = []
        if not reset and exists(path):
            self._load()
        else:
            self._save()

    def _save(self):
        with open(self._path, "w") as file:
            obj = {"entries": [e.__dict__ for e in self._entries]}
            json.dump(obj, file)

    def _load(self):
        with open(self._path, "r") as file:
            obj = json.load(file)
            self._entries = [TrainingLogEntry(**dct) for dct in obj["entries"]]

    @property
    def best_entry(self) -> TrainingLogEntry:
        return min(self._entries, key=lambda e: e.validation_equal_error_rate)

    def __len__(self) -> int:
        return len(self._entries)

    def add_entry(self, entry: TrainingLogEntry):
        self._entries.append(entry)
        self._save()


def _train(
    model: torch.nn.Module,
    loss_fun: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_set: Dataset,
) -> float:
    """
    Trains the model for one epoch.

    Returns
        - overall average epoch loss
        - a dict with the epoch loss of individual loss components
    """
    metric = torchmetrics.classification.MulticlassAccuracy(
        num_classes=train_set.num_subjects
    ).to(device=get_device())

    train_dataloader = torch.utils.data.DataLoader(
        train_set, **get_dataloader_args(train=True)
    )
    model.train()  # Outputs minutia maps and logits
    epoch_loss = 0
    loss_fun.reset_recorded_loss()
    for vals in tqdm.tqdm(train_dataloader):
        fp_imgs, minu_map_tpl, fp_labels = vals
        minu_maps, minu_map_weights = minu_map_tpl
        fp_imgs = fp_imgs.to(device=get_device())
        fp_labels = fp_labels.to(device=get_device())
        minu_maps = minu_maps.to(device=get_device())
        minu_map_weights = minu_map_weights.to(device=get_device())
        # Forward pass
        optimizer.zero_grad()
        output: DeepPrintTrainingOutput = model(fp_imgs)
        loss = loss_fun.forward(
            output=output,
            labels=fp_labels,
            minutia_maps=minu_maps,
            minutia_map_weights=minu_map_weights,
        )
        # Backward pass
        loss.backward()
        optimizer.step()

        # Record accuracy and loss
        epoch_loss += float(loss) * fp_labels.shape[0]
        if output.combined_logits is not None:
            logits = output.combined_logits
        elif output.minutia_logits is None:
            logits = output.texture_logits
        elif output.texture_logits is None:
            logits = output.minutia_logits
        else:
            logits = output.texture_logits + output.minutia_logits
        metric(logits, fp_labels)

    mean_loss = epoch_loss / len(train_set)
    multiclass_accuracy = float(metric.compute())
    return mean_loss, loss_fun.get_recorded_loss(), multiclass_accuracy


def _validate(
    model: torch.nn.Module,
    validation_set: Dataset,
    benchmark: VerificationBenchmark,
) -> float:
    """
    Validates the model.

    Returns equal error rate
    """
    texture_embeddings, minutia_embeddings = extract_embeddings(model, validation_set)
    embeddings = EmbeddingLoader.combine_if_both_exist(
        texture_embeddings, minutia_embeddings
    )

    matcher = CosineSimilarityMatcher(embeddings)
    result: VerificationResult = benchmark.run(matcher, save=False)
    return result.get_equal_error_rate()


def train_model(
    fingerprints: Dataset,
    minutia_maps: Dataset,
    labels: Dataset,
    validation_fingerprints: Dataset,
    validation_benchmark: VerificationBenchmark,
    model: torch.nn.Module,
    loss: torch.nn.Module,
    num_epochs: int,
    out_dir: str,
) -> None:
    """
    Trains model for num_iter and saves results (training log and model parameters)

    Automatically loads model parameters from "model.pyt" file if exists
    """
    print(f"Using device {get_device()}")
    # Create output directory and log file
    best_model_path = get_best_model_file(out_dir)
    model_path = get_newest_model_file(out_dir)
    log = TrainingLog(join(out_dir, "log.json"))

    model = model.to(device=get_device())
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    loss = loss.to(device=get_device())
    if exists(model_path):
        print(f"Loaded existing model from {model_path}")
        load_model_parameters(model_path, model, loss, optimizer)
    else:
        print(f"No model file found at {model_path}")

    training_set = Dataset.zip(fingerprints, minutia_maps, labels)

    for epoch in range(len(log) + 1, num_epochs + 1):
        print(f"\n\n --- Starting Epoch {epoch} of {num_epochs} ---")
        # Train
        print("\nTraining:")
        train_loss, loss_stats, accuracy = _train(model, loss, optimizer, training_set)
        print(f"Average Loss: {train_loss}")
        print(f"Multiclass accuracy: {accuracy}")

        save_model_parameters(model_path, model, loss, optimizer)

        if validation_fingerprints is None:
            # Use training accuracy as validation accuracy
            validation_eer = accuracy
        else:
            # Validate
            print("\nValidation:")
            validation_eer = _validate(model, validation_fingerprints, validation_benchmark)
            print(f"Equal Error Rate: {validation_eer}\n")


        # Log and determine if new model is best model
        entry = TrainingLogEntry(
            epoch, train_loss, loss_stats, accuracy, validation_eer
        )
        log.add_entry(entry)
        print(entry)

        if validation_eer <= log.best_entry.validation_equal_error_rate:
            shutil.copyfile(model_path, best_model_path)
