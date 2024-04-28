import sys, os
import argparse
from subprocess import Popen, PIPE
import time
from datetime import date
from itertools import product

from hydra import compose, initialize
from omegaconf import OmegaConf
import tempfile


def outer_product(sweep):
    # Extract argument names and their possible values
    keys, values = zip(*[(key, val[1]) for key, val in sweep.items()])
    # Compute the Cartesian product of the values
    product_combinations = product(*values)
    # Create a dictionary for each combination
    result = []
    for combination in product_combinations:
        combination_dict = {
            sweep[key][0]: value for key, value in zip(keys, combination)
        }
        result.append(combination_dict)
    return result


def check_for_done(l):
    for i, p in enumerate(l):
        if p.poll() is not None:
            return True, i
    return False, False


parser = argparse.ArgumentParser(description="Training arguments")
# Optimizer
<<<<<<< HEAD
# python3 scripts/pipe.py --num_runs 1 --gpu_ids ??? 
=======
# python3 scripts/pipe.py --num_runs 1 --gpu_ids 0 --wandb --print --logrun --num_epochs 500 --group_name
>>>>>>> a098429 (ablations conflicts solved)
parser.add_argument("--num_runs", default=1, type=int, help="Number of runs to run")
parser.add_argument(
    "--gpu_ids",
    help="delimited list input",
    type=lambda s: [int(item) for item in s.split(",")],
)
parser.add_argument("--wandb", action="store_true", help="To use wandb logging or not", default=True)
parser.add_argument("--logrun", action="store_true", help="save the model orn ot", default=True)
parser.add_argument("--print", action="store_true", help="to print the outputs or not", default=True)
# parser.add_argument(
#     "--num_epochs", default=500, type=int, help="Number of epochs to learn"
# )
# parser.add_argument("--group_name", default=None, type=str, help="Group Name")

args = parser.parse_args()

print(args.gpu_ids)

CFGs_DIR = "pipe_cfgs"
# create directory if it doesn't exist
os.makedirs(CFGs_DIR, exist_ok=True)

NUM_RUNS = args.num_runs
GPU_IDS = args.gpu_ids
NUM_GPUS = len(GPU_IDS)
USE_WANDB = args.wandb
LOG_RUN = args.logrun
PRINT = args.print

counter = 0


NUM_EPOCHS = 200
GROUP_NAME = "final2"
sweep = dict(
<<<<<<< HEAD
    d_model=("model.params.d_model", [32,64]),
    # n_layers=("model.params.num_layers", [4]),
    K=("model.params.K", [4]),
    lr=("model.optimizer.lr", [0.001, 0.002]),
    
    model=("model", ["GREDMamba","gine"]),
=======
    lr=("model.optimizer.lr", [0.001]),
    n_layers=("model.params.num_layers", [10]), #, 4, 2
    model=("model", ["gine"]), # , "gine-mamba", "GREDMamba","GREDMamba-mamba"
>>>>>>> a098429 (ablations conflicts solved)
    dataset=("dataset", ["zinc"]),
    # dataset=("dataset", ["zinc","peptides-struct"]),
    # evaluate_frequency=('evaluate_frequency', [5]),
    batch_size=("model.batch_size", [256]),
    warmup_epochs=("model.warmup_epochs", [1]),
<<<<<<< HEAD
    # seed=("seed", [33, 36]),
=======
    seed=("seed", [33]), #36
>>>>>>> a098429 (ablations conflicts solved)
    
    
    d_state=("model.params.d_state", [16]),
    

    scheduler=('model.scheduler', ['cosine']),
)


sweep_product = list(outer_product(sweep))
length = len(sweep_product)
print(f"Number of runs that will be executed: {length}")
# print(sweep_product)


wandb_group = GROUP_NAME

wandb_tags_original = []


procs = list()
gpu_queue = list()
gpu_use = list()

for i in range(NUM_RUNS):
    gpu_queue.append(GPU_IDS[i % NUM_GPUS])


cfgs_paths = []
for run in sweep_product:
    overrides = []
    for key, value in run.items():
        overrides.append(f"{key}={value}")

    overrides.extend(
        [
            f"print={PRINT}",
            f"use_wandb={USE_WANDB}",
            f"log_model={LOG_RUN}",
            f"num_epochs={NUM_EPOCHS}",
            f"wandb.group={wandb_group}",
        ]
    )

    wandb_tags = wandb_tags_original

    if len(wandb_tags) > 0:
        overrides.append(f"'wandb.tags={wandb_tags}'")
    else:
        pass

    with initialize(version_base=None, config_path="../configs"):
        # print(overrides)
        time.sleep(1)
        cfg = compose(config_name="config", overrides=overrides)
        # resolve
        OmegaConf.resolve(cfg)
        # print(OmegaConf.to_yaml(cfg))
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=CFGs_DIR, suffix=".tmp"
    ) as f:
        OmegaConf.save(cfg, f.name)
        cfgs_paths.append(f.name)

for path in cfgs_paths:
    gpu_id = gpu_queue.pop(0)
    gpu_use.append(gpu_id)
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python train_precomposed.py --cfg_path {path}"
    print(cmd)

    procs.append(Popen(cmd, shell=True))

    time.sleep(3)

    counter += 1

    if len(procs) == NUM_RUNS:
        wait = True

        while wait:
            done, num = check_for_done(procs)

            if done:
                procs.pop(num)
                gpu_queue.append(gpu_use.pop(num))
                wait = False
            else:
                time.sleep(3)

        print("\n \n \n \n --------------------------- \n \n \n \n")
        print(f"{date.today()} - {counter} runs completed")
        sys.stdout.flush()
        print("\n \n \n \n --------------------------- \n \n \n \n")

for p in procs:
    p.wait()
procs = []
