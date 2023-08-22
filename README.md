## Disclaimer: There will be major update with refactoring and examples of how to use the software until 17.9.

Implementation of Deep Print, a Deep Learning based fixed-length Fingerprint Representation extractor, and various experiments with such architectures. For scientific background see:

[Benchmarking fixed-length Fingerprint Representations across different Embedding Sizes and Sensor Types](https://arxiv.org/abs/2307.08615)

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── poses          <- Folder for poses of a corresponding fingerprint dataset.
    │   ├── embeddings     <- Output folder for fingerprint embeddings.
    │   └── fingerprints   <- Folder with fingerprint datasets.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports            <- Contains folders with the full benchmark results of all trained models
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        │
        ├── benchmarks     <- Implementation of biometric performance benchmarks (verification, identification)
        │   │                 Includes matching, de/serialization of results, calculation of various metrics...
        │   └── generate_benchmarks.py <- Script to generate the json files that describe the verification and
        │                                 identification benchmark for each dataset
        │
        ├── data           <- Code to manage datasets of fingerprints, poses, embeddings etc.
        │
        ├── models         <- Model architectures and training scripts for individual architectures
        │
        ├── models         <- Model architectures and loss functions
        │
        │
        │
        ├── visualization  <- Contains functions for visualization and graphical debugging
        │
        ├── generate_embeddings.py  <- Script to generate the embeddings of all test sets
        ├── benchmark_embeddings.py <- Script to run the benchmarks for all test sets using the generated embeddings
        ├── config.py      <- Contains experimental settings (training, validation, test sets etc.) that are
        │                     imported by many of the runnable scripts.
        ├── paths.py       <- Contains functions that describe the folder structure and path scheme to keep this responsibility 
        │                     out of individual classes and scripts as much as possible
        │
        └── torch_helpers.py  <- Model saving/loading, embedding generation and some utility functions related to GPUs
     
     



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


--------

### Credit to other authors:

Remi Cadene - Inception v4 pytorch implementation - src/models/Inceptionv4.py - https://github.com/Cadene/pretrained-models.pytorch
Dong Chengdong Hang Zhou - iso-19794-2 fingerprint template encoder and decoder - https://github.com/DongChengdongHangZhou/iso-19794-2-decoder-encoder
