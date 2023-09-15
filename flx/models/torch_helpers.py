import os
import time

import torch


CUDA_DEVICE = 0
TRAIN_ON_A_100 = True


def get_dataloader_args(train: bool) -> dict:
    batch_size = 16
    if not train:
        batch_size *= 2  # More memory available without gradients
    if not torch.cuda.is_available():
        return {
            "batch_size": batch_size,
            "shuffle": train,
            "num_workers": 4,
            "prefetch_factor": 1,
        }
    if TRAIN_ON_A_100:  # Use 40GB graphics ram by preloading to pinned memory
        return {
            "batch_size": batch_size,
            "shuffle": train,
            "num_workers": 16,
            "prefetch_factor": 2,
            "pin_memory": True,
            "pin_memory_device": f"cuda:{CUDA_DEVICE}",
        }
    return {
        "batch_size": batch_size,
        "shuffle": train,
        "num_workers": 4,
        "prefetch_factor": 1,
        "pin_memory": True,
        "pin_memory_device": f"cuda:{CUDA_DEVICE}",
    }


def get_device() -> str:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{CUDA_DEVICE}")
    return torch.device("cpu")


def save_model_parameters(
    full_param_path: str,
    model: torch.nn.Module,
    loss: torch.nn.Module,
    optim: torch.optim.Optimizer,
) -> None:
    """
    Tries to save the parameters of model and optimizer in the given path
    """
    try:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "loss_state_dict": loss.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
            },
            full_param_path,
        )
    except KeyboardInterrupt:
        print("\n>>>>>>>>> Model is being saved! Will exit when done <<<<<<<<<<\n")
        save_model_parameters(full_param_path, model, optim)
        time.sleep(10)
        raise KeyboardInterrupt()


def load_model_parameters(
    full_param_path: str,
    model: torch.nn.Module,
    loss: torch.nn.Module,
    optim: torch.optim.Optimizer,
) -> None:
    """
    Tries to load the parameters stored in the given path
    into the given model and optimizer.
    """
    if not os.path.exists(full_param_path):
        raise FileNotFoundError(f"Model file {full_param_path} did not exist.")
    checkpoint = torch.load(full_param_path, map_location=get_device())
    model.load_state_dict(checkpoint["model_state_dict"])
    if loss is not None:
        loss.load_state_dict(checkpoint["loss_state_dict"])
    if optim is not None:
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
