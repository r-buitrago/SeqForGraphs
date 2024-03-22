from train import _main
from omegaconf import OmegaConf
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cfg_path", type=str, required=True)

    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg_path)
    _main(cfg)
