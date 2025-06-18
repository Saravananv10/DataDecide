from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer

from .config import OptimizerConfig, SchedulerConfig, TrainConfig

__all__ = ["build_optimizer", "build_scheduler"]


def build_optimizer(config: TrainConfig, model: torch.nn.Module) -> Optimizer:
    """
    Build an optimizer.
    """
    optimizer_config = config.optimizer
    optimizer_name = optimizer_config.name.lower()

    # Get parameters that require gradients.
    params = list(model.parameters())
    if not params:
        raise ValueError("Cannot build optimizer because model has no parameters")

    # Filter out parameters that don't require gradients.
    trainable_params = [p for p in params if p.requires_grad]
    if not trainable_params:
        raise ValueError(
            "Cannot build optimizer because model has no trainable parameters. "
            "Perhaps you called `model.eval()` or `model.requires_grad_(False)`?"
        )

    # Get weight decay parameters.
    # We don't want to apply weight decay to bias terms or layer norm parameters.
    no_decay: Set[torch.nn.Parameter] = set()
    decay: Set[torch.nn.Parameter] = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if not p.requires_grad:
                continue
            # NOTE: biases and layer norm parameters are generally not decayed.
            # For this specific model, we also don't want to decay the
            # query, key, and value projection parameters in the attention layers.
            # This is because they are already regularized by the attention dropout.
            # We also don't want to decay the token embeddings.
            if pn.endswith(".bias") or pn.endswith("_norm.weight") or ".ln" in pn or ".norm" in pn:
                no_decay.add(p)
            elif ".embed_tokens." in pn:  # Don't decay token embeddings.
                no_decay.add(p)
            # elif ".attn.q_proj." in pn or ".attn.k_proj." in pn or ".attn.v_proj." in pn:
            #     # Don't decay query, key, and value projection parameters.
            #     # This is because they are already regularized by the attention dropout.
            #     no_decay.add(p)
            else:
                decay.add(p)

    # Sanity check.
    # TODO: this might not be true if there are shared parameters.
    # if len(decay) + len(no_decay) != len(trainable_params):
    #     raise ValueError(
    #         f"Number of decay parameters ({len(decay)}) and no_decay parameters ({len(no_decay)}) "
    #         f"does not equal number of trainable parameters ({len(trainable_params)})"
    #     )

    # Create optimizer parameter groups.
    param_groups = [
        {"params": list(decay), "weight_decay": optimizer_config.weight_decay},
        {"params": list(no_decay), "weight_decay": 0.0},
    ]

    # Build optimizer.
    if optimizer_name == "adamw":
        return optim.AdamW(
            param_groups,
            lr=optimizer_config.learning_rate,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            **optimizer_config.args,
        )
    elif optimizer_name == "adam":
        return optim.Adam(
            param_groups,
            lr=optimizer_config.learning_rate,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
            **optimizer_config.args,
        )
    elif optimizer_name == "sgd":
        return optim.SGD(
            param_groups,
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            **optimizer_config.args,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config.name}")


def build_scheduler(config: TrainConfig, optimizer: Optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Build a learning rate scheduler.
    """
    scheduler_config = config.scheduler
    scheduler_name = scheduler_config.name.lower()

    # Resolve t_warmup and t_max.
    if isinstance(scheduler_config.t_warmup, str):
        t_warmup = _parse_scheduler_time(scheduler_config.t_warmup, config)
    else:
        t_warmup = scheduler_config.t_warmup
    if isinstance(scheduler_config.t_max, str):
        t_max = _parse_scheduler_time(scheduler_config.t_max, config)
    else:
        t_max = scheduler_config.t_max

    # Build scheduler.
    if scheduler_name == "cosine_with_warmup":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_warmup,
            T_mult=1,  # TODO: make this configurable
            eta_min=scheduler_config.alpha_f * config.optimizer.learning_rate,
            **scheduler_config.args,
        )
    elif scheduler_name == "linear_with_warmup":
        # TODO: this is not actually a linear scheduler with warmup.
        # It's a linear scheduler that warms up to the initial learning rate
        # and then decays to the final learning rate.
        # We should probably rename this to something like "linear_warmup_linear_decay".
        # Or, we could just use `transformers.get_linear_schedule_with_warmup`.
        # For now, we'll just use this simple implementation.
        def lr_lambda(current_step: int):
            if current_step < t_warmup:
                return float(current_step) / float(max(1, t_warmup))
            return max(
                0.0,
                scheduler_config.alpha_f
                + (1.0 - scheduler_config.alpha_f)
                * (t_max - current_step)
                / (t_max - t_warmup),
            )

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, **scheduler_config.args)
    elif scheduler_name == "constant_with_warmup":

        def lr_lambda(current_step: int):
            if current_step < t_warmup:
                return float(current_step) / float(max(1, t_warmup))
            return 1.0

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, **scheduler_config.args)
    elif scheduler_name == "none" or scheduler_name == "constant":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_config.name}")


def _parse_scheduler_time(time_str: str, config: TrainConfig) -> int:
    """
    Parse a scheduler time string like "1000ba" or "0.1ep".
    """
    if time_str.endswith("ba"):
        return int(time_str[:-2])
    elif time_str.endswith("ep"):
        # TODO: this is not quite right.
        # It assumes that the number of batches per epoch is constant,
        # which is not necessarily true if the dataset size is not divisible
        # by the batch size.
        # We should probably just use the total number of batches instead.
        # For now, we'll just use this simple implementation.
        if config.data.data_loader_args is None:
            raise ValueError(
                "Cannot parse scheduler time string with 'ep' suffix "
                "because data_loader_args is not set in data config"
            )
        batches_per_epoch = len(config.data.paths) // config.global_train_batch_size  # type: ignore
        return int(float(time_str[:-2]) * batches_per_epoch)
    else:
        raise ValueError(f"Invalid scheduler time string: {time_str}")
