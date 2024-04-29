import sys, os
import numpy as np
import hydra
from hydra.utils import instantiate
import wandb
from omegaconf import OmegaConf, DictConfig

import pickle
import random
import torch
from timeit import default_timer

from utils.log_utils import get_logger, add_file_handler
import logging
from utils.evaluator import DummyEvaluator, eval_to_print, eval_to_wandb
from utils.utils import count_params
from utils.scheduler import WarmupScheduler
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

log = get_logger(__name__, level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataset(args, batch_size):
    train_data = instantiate(args.dataset.params_train)
    test_data = instantiate(args.dataset.params_test)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=args.num_workers, shuffle=False
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, num_workers=args.num_workers, shuffle=False
    )
    return train_loader, test_loader

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=3):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


def learning_step(
    args,
    model,
    batch,
    loss_type,
    is_training=True,
    evaluator=None,
    batch_start=0,
    calculate_embedding_diff=False,
    calculate_embedding=False
):
    batch.to(device)
    loss_fn = get_loss_value(loss_type)


    # precomputed_masks_path = args.dataset.params.precomputed_masks_path
    # if precomputed_masks_path is not None:
    #     dist_mask = DIST_MASKS[
    #         batch_start : batch_start + batch.num_graphs
    #     ].to(
    #         device
    #     )  # (B, K+1, max_nodes, max_nodes)
    # else:
    #     dist_mask = None
    
    out, embedding_diff, embeddings = model(
        batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch, batch.get("dist_mask", None), calculate_embedding_diff, calculate_embedding
    )
    loss = loss_fn(out.squeeze(), batch.y.float())
    
    if evaluator is not None:
        evaluator.evaluate(out.squeeze(), batch.y.float())
    return loss, embedding_diff, embeddings



def get_loss_value(loss_type):
    if loss_type == "mse":
        return torch.nn.MSELoss(reduction="sum")
    elif loss_type == "bce":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_type == "ce":
        return torch.nn.CrossEntropyLoss()
    elif loss_type == "focal":
        return FocalLoss()
    else:
        raise ValueError(f"Loss type {loss_type} not supported")


def train(
    args,
    train_loader,
    model,
    optimizer,
    loss_type,
    scheduler=None,
    evaluator=DummyEvaluator,
):
    model.train()
    t1 = default_timer()
    batch_timer = default_timer()
    for i, batch in enumerate(train_loader):
        loss, _, _ = learning_step(
            args,
            model,
            batch,
            loss_type=loss_type,
            evaluator=evaluator,
            batch_start=i,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        log.debug(f"Training batch time: {default_timer() - batch_timer}")
        batch_timer = default_timer()
    t2 = default_timer()
    return model, t2 - t1


@torch.no_grad()
def test(args, test_loader, model, loss_type, evaluator=DummyEvaluator):
    model.eval()
    t1 = default_timer()
    batch_timer = default_timer()
    for i, batch in enumerate(test_loader):
        loss, _, _ = learning_step(
            args,
            model,
            batch,
            loss_type=loss_type,
            evaluator=evaluator,
            is_training=False,
            batch_start=i,
        )
        log.debug(f"Evaluation batch time: {default_timer() - batch_timer}")
        batch_timer = default_timer()
    t2 = default_timer()
    return t2 - t1


@torch.no_grad()
def oversmoothing_statistics(args, test_loader, model, loss_type, evaluator=DummyEvaluator):
    model.eval()
    t1 = default_timer()
    batch_timer = default_timer()

    batch_embedding_diffs = 0
    batches = 0
    
    for i, batch in enumerate(test_loader):

        loss, layer_embedding_diff, _ = learning_step(
            args,
            model,
            batch,
            loss_type=loss_type,
            evaluator=evaluator,
            is_training=False,
            batch_start=i,
            calculate_embedding_diff=True
        )

        batch_embedding_diffs += layer_embedding_diff
        batches += 1

        log.debug(f"Evaluation batch time: {default_timer() - batch_timer}")
        batch_timer = default_timer()

    batch_embedding_diffs = batch_embedding_diffs / batches

    log.info(f'Oversmoothing Analysis, by Layer: {batch_embedding_diffs}')
    
    t2 = default_timer()
    return t2 - t1, batch_embedding_diffs 


def oversquashing_statistics(
    args,
    train_loader,
    model,
    optimizer,
    loss_type,
    scheduler=None,
    evaluator=None,
):
    model.train()
    t1 = default_timer()
    batch_timer = default_timer()

    sensitivity_matrix_all = 0
    batches = 0

    for i, batch in enumerate(train_loader):
        loss, _, batch_embedding = learning_step(
            args,
            model,
            batch,
            loss_type=loss_type,
            evaluator=evaluator,
            batch_start=i,
            calculate_embedding=True
        )
        optimizer.zero_grad()

        
        layers_, nodes_, dim_, = batch_embedding.shape
        sensitivity_matrix = torch.zeros(layers_)

        for node in random.sample(range(nodes_), 100):
            for f in range(0, dim_):
                batch_embedding[-1][node, f].backward(retain_graph=True)

                for hidden_layer in range(layers_):
                    if batch_embedding[hidden_layer].grad != None:
                        sensitivity_matrix[hidden_layer] += torch.norm(batch_embedding[hidden_layer].grad, p=1).cpu().numpy()
                        batch_embedding[hidden_layer].grad = None

        sensitivity_matrix_all += sensitivity_matrix
        batches += 1

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        log.debug(f"Training batch time: {default_timer() - batch_timer}")
        batch_timer = default_timer()

    sensitivity_matrix_all = sensitivity_matrix_all / batches
    log.info(f'Oversquashing Analysis, by Layer: {sensitivity_matrix_all}')

    t2 = default_timer()
    return t2 - t1, sensitivity_matrix_all

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(args: DictConfig):
    _main(args)


def _main(args: DictConfig):
    OmegaConf.update(args.model.params, "feature_dim", args.dataset.feature_dim)
    OmegaConf.update(args.model.params, "edge_dim", args.dataset.edge_dim)
    OmegaConf.update(args.model.params, "classes", args.dataset.classes)
    OmegaConf.update(args.model.params, "embed_type", args.dataset.embed_type)

    if not args.print:
        log.setLevel(logging.WARNING)
    log.info(OmegaConf.to_yaml(args))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.num_threads)

    if args.use_wandb:
        log.enable_wandb()
        wandb_config = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
        if args.wandb.tags is not None:
            run = wandb.init(
                project=args.wandb.project,
                group=args.wandb.group,
                name=args.wandb.name,
                config=wandb_config,
                tags=list(args.wandb.tags),
            )
        else:
            run = wandb.init(
                project=args.wandb.project,
                group=args.wandb.group,
                name=args.wandb.name,
                config=wandb_config,
            )

    if args.log_model:
        model_folder_path = os.path.join(
            args.workdir.root,
            args.workdir.name,
        )
        os.makedirs(model_folder_path, exist_ok=True)
    train_loader, test_loader = get_dataset(args, batch_size=args.model.batch_size)

    _data = next(iter(test_loader))
    print(f"Dataset shape: {_data}")

    model = instantiate(args.model.params)
    model.to(device)

    log.info(count_params(model))
    total_params = sum(p.numel() for p in model.parameters())
    total_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total parameters = {total_params} require_grad {total_grad_params}")
    if args.use_wandb:
        run.summary["total_parameters"] = total_params
        run.summary["total_grad_parameters"] = total_grad_params

    optimizer = instantiate(args.model.optimizer, params=model.parameters())

    iterations = args.num_epochs * len(train_loader.dataset) // args.model.batch_size
    # iterations = args.num_epochs
    if args.model.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iterations
        )
    elif args.model.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.model.step_size
            * len(train_loader.dataset)
            // args.model.batch_size,
            gamma=args.model.gamma,
        )
    else:
        scheduler = None

    if scheduler is not None and args.model.warmup_epochs > 0:
        scheduler = WarmupScheduler(
            optimizer,
            warmup_steps=args.model.warmup_epochs
            * len(train_loader.dataset)
            // args.model.batch_size,
            base_scheduler=scheduler,
        )

    loss_type = args.loss.type
    train_evaluator = instantiate(
        args.evaluator.train, loss_fn=get_loss_value(loss_type)
    )
    test_evaluator = instantiate(args.evaluator.test, loss_fn=get_loss_value(loss_type))

    for epoch in range(args.num_epochs):
        model, train_time = train(
            args,
            train_loader,
            model,
            optimizer,
            loss_type,
            scheduler,
            evaluator=train_evaluator,
        )
        if epoch % args.evaluate_frequency == 0:
            test_time = test(
                args, test_loader, model, loss_type, evaluator=test_evaluator
            )

        if epoch % args.oversmoothing_frequency == 0:
            test_time, batch_embedding_diffs = oversmoothing_statistics(
                args, test_loader, model, loss_type, evaluator=test_evaluator
            )

        if epoch % args.oversquashing_frequency == 0:
            test_time, sensitivity_matrix_all = oversquashing_statistics(
                args,
                train_loader,
                model,
                optimizer,
                loss_type,
                scheduler,
                evaluator=train_evaluator
            )

        else:
            test_time = 0.0

        train_evaluation = train_evaluator.get_evaluation()
        train_evaluator.reset()
        test_evaluation = test_evaluator.get_evaluation()
        test_evaluator.reset()

        train_msg = eval_to_print(train_evaluation, is_train=True)
        test_msg = eval_to_print(test_evaluation, is_train=False)
        log.info(
            f"Epoch {epoch}"
            + train_msg
            + test_msg
            + " | train_time "
            + f"{train_time:.2f}"
            + " | test_time "
            + f"{test_time:.2f}"
        )

        log.wandb(eval_to_wandb(train_evaluation, is_train=True), step=epoch)
        log.wandb(eval_to_wandb(test_evaluation, is_train=False), step=epoch)
        log.wandb(
            {"time/train_time": train_time, "time/test_time": test_time}, step=epoch
        )

        if args.log_model:
            if epoch % args.log_frequency == 0:
                checkpoint_path = os.path.join(model_folder_path, "ckpt.pt")
                torch.save(model.state_dict(), checkpoint_path)
                cfg_path = os.path.join(model_folder_path, "cfg.yaml")
                with open(cfg_path, "w") as f:
                    f.write(OmegaConf.to_yaml(args))


                log.info(
                    f"Saving model to {checkpoint_path}"
                )


if __name__ == "__main__":

    main()
