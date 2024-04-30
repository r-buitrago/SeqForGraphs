
### About

SeqForGraphs is an implementation of GRED-Mamba benchmarked on Zinc and Peptide datasets.

### Environment
A list of the needed packages is given in `requirements.txt`. All the experiments were run with Python 3.10.14.

### Running Instructions

This repository uses hydra to configure experiments. The current config files are designed to run GREDMamba on the ZINC dataset, so running train.py directly would run such experiment. All the experiments can be run by changing the config files, in particular setting model to one of `"gine"`, `"GREDMamba"` or `"GREDMamba-Mamba"` and dataset to `"zinc"` and `"peptides-struct"` allows running all experiments.

> ***NOTE***: Before runing train.py, the notebooks in `notebooks/precomputed_dist_masks.ipynb` and `notebooks/precomputed_dist_masks_zinc.ipynb`, must be run which precomputes the k-hops distance for each node from the target node and save them to a data folder. The experiments without precomputed distance masks can be run by changin the `"use_distance_masks"` variables in `zinc.yaml` and `peptides-struct.yaml`, but this is not recommended since training would be very slow. When running `notebooks/precomputed_dist_masks.ipynb` or `notebooks/precomputed_dist_masks_zinc.ipynb`, the `"SPLIT"` and `"N_SAMPLES_MAX"` must be set to generate distance masks for the splits `"train"` and `"val"` and a given number of samples. This number of samples must be the same as the one set in `zinc.yaml` and `peptides-struct.yaml`.

### Hyperparameter Sweep Instructions

Example of pipe run:
`python3 scripts/pipe.py --num_runs 1 --gpu_ids 0 --wandb --print --logrun --num_epochs 21 --group_name test`

Specify hyperparameters in scripts/pipe.py:

`sweep = dict(
    d_model=("model.params.d_model", [32,64]),
    # n_layers=("model.params.num_layers", [4]),
    K=("model.params.K", [4]),
    lr=("model.optimizer.lr", [0.001, 0.002]),
)`

Specify default configuration parameters in configs/config.yaml


