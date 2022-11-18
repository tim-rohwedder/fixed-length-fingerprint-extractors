from os.path import join, dirname, abspath, exists, isdir
from os import makedirs, remove, removedirs
from shutil import rmtree
from typing import Callable

BASE_DIR = dirname(dirname(dirname(abspath(__file__))))
MODELS_DIR = join(BASE_DIR, "models")
REPORTS_DIR = join(BASE_DIR, "reports")
BENCHMARKS_DIR = join(BASE_DIR, "data", "benchmarks")
FINGERPRINTS_DIR = join(BASE_DIR, "data", "fingerprints")
EMBEDDINGS_DIR = join(BASE_DIR, "data", "embeddings")
POSES_DIR = join(BASE_DIR, "data", "poses")
DEBUG_DIR = join(BASE_DIR, "debug")


def created_parent_dir(pathfun: Callable) -> str:
    def make_parent_and_return(*args, **kwargs):
        path = pathfun(*args, **kwargs)
        if not exists(dirname(path)):
            makedirs(dirname(path))
        return path

    return make_parent_and_return


def created_dir(pathfun: Callable) -> str:
    def make_parent_and_return(*args, **kwargs):
        path = pathfun(*args, **kwargs)
        if not exists(path):
            makedirs(path)
        return path

    return make_parent_and_return


def remove_path(path: str) -> None:
    """
    Removes the object, no matter if it is a file or directory. Does nothing if it
    does not exist.
    """
    try:
        remove(path)
        removedirs(dirname(path))  # Remove directory if now empty
        return
    except OSError:
        pass
    try:
        rmtree(path, ignore_errors=True)
        removedirs(dirname(path))  # Remove directory if now empty
        return
    except OSError:
        pass


# Dataset paths
def get_fingerprint_dataset_path(dataset_name: str) -> str:
    return join(FINGERPRINTS_DIR, dataset_name)


@created_dir
def get_model_dir(model_name: str) -> str:
    return join(MODELS_DIR, model_name)


@created_parent_dir
def get_best_model_file(model_dir: str) -> str:
    return join(model_dir, "best_model.pyt")


@created_parent_dir
def get_newest_model_file(model_dir: str) -> str:
    return join(model_dir, "model.pyt")


# Embedding paths
@created_dir
def get_generated_embeddings_dir(model_name: str, dataset_name: str) -> str:
    return join(EMBEDDINGS_DIR, model_name, dataset_name)


@created_dir
def get_texture_embedding_dataset_dir(embedding_base_dir: str) -> str:
    return join(
        embedding_base_dir,
        "texture",
    )


@created_dir
def get_minutia_embedding_dataset_dir(embedding_base_dir: str) -> str:
    return join(
        embedding_base_dir,
        "minutia",
    )


@created_dir
def get_reweighted_embedding_dataset_dir(embedding_base_dir: str) -> str:
    return join(
        embedding_base_dir,
        "reweighted",
    )


# Pose paths
@created_parent_dir
def get_pose_dataset_path(dataset_name: str) -> str:
    return join(POSES_DIR, f"{dataset_name}.json")


# Benchmark paths
@created_parent_dir
def get_verification_benchmark_file(testset_name: str) -> str:
    return join(BENCHMARKS_DIR, "verification", testset_name + ".json")


@created_parent_dir
def get_open_set_benchmark_file(testset_name: str) -> str:
    return join(BENCHMARKS_DIR, "identification_open_set", testset_name + ".json")


@created_parent_dir
def get_closed_set_benchmark_file(testset_name: str) -> str:
    return join(BENCHMARKS_DIR, "identification_closed_set", testset_name + ".json")


# Result and report paths
@created_parent_dir
def get_benchmark_results_dir(model_name: str, testset_name: str) -> str:
    return join(REPORTS_DIR, model_name, testset_name)


@created_parent_dir
def get_verification_benchmark_results_file(model_name: str, testset_name: str) -> str:
    return join(
        get_benchmark_results_dir(model_name, testset_name), "verification.json"
    )


@created_dir
def get_closed_set_benchmark_results_dir(model_name: str, testset_name: str) -> str:
    return join(
        get_benchmark_results_dir(model_name, testset_name),
        "identification_closed_set",
    )


@created_dir
def get_open_set_benchmark_results_dir(model_name: str, testset_name: str) -> str:
    return join(
        get_benchmark_results_dir(model_name, testset_name),
        "identification_open_set",
    )


@created_dir
def get_figures_dir(testset_name: str, *subdir_names: list[str]) -> str:
    return join(REPORTS_DIR, "figures", testset_name, *subdir_names)


@created_parent_dir
def get_debug_file(*args: list[str]) -> str:
    return join(DEBUG_DIR, *args)
