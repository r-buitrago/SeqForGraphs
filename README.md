
### About

SeqForGraphs is an implementation of GRED-Mamba benchmarked on Zinc and Peptide datasets.

### Running Instructions

python3 train.py

### Hyperparameter Sweep Instructions

python3 scripts/pipe.py --num_runs 1 --gpu_ids 0 --wandb --print --logrun --num_epochs 21 --group_name test

Specify hyperparameters in scripts/pipe.py:

`sweep = dict(
    d_model=("model.params.d_model", [32,64]),
    # n_layers=("model.params.num_layers", [4]),
    K=("model.params.K", [4]),
    lr=("model.optimizer.lr", [0.001, 0.002]),
)`

Specify default configuration parameters in configs/config.yaml

> ***NOTE***: Training can be sped-up by running notebooks/precomputed_dist_masks.ipynb, which precomputes the k-hops distance for each node from the target node.
