from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from .aliases import PathOrStr
from .exceptions import OLMoConfigurationError

__all__ = [
    "Activation",
    "LayerNormBase",
    "LowPrecisionLayerNorm",
    "RMSNorm",
    "get_global_rank",
    "get_world_size",
    "barrier",
    "all_reduce",
    "all_gather",
    "broadcast",
    "reduce_sum",
    "is_distributed",
    "get_fsdp_wrap_module_name",
    "get_fsdp_wrap_module_type",
    "get_fsdp_device_mesh",
    "get_fsdp_world_size",
    "get_fsdp_rank",
    "get_fsdp_strategy",
    "get_fsdp_auto_wrap_policy",
    "get_fsdp_cpu_offload",
    "get_fsdp_mixed_precision_policy",
    "get_fsdp_backward_prefetch_policy",
    "get_fsdp_sharding_strategy",
    "get_fsdp_limit_all_gathers",
    "get_fsdp_use_orig_params",
    "fsdp_model_state_dict_rank0",
    "fsdp_full_optim_state_dict_rank0",
    "is_rank0",
    "TorchCompileConfig",
    "maybe_compile",
]


class Activation(nn.Module):
    """
    A wrapper for activation functions.
    This is used to allow specifying activation functions as strings in the config.
    """

    def __init__(self, activation_type: str):
        super().__init__()
        self.activation_type = activation_type.lower()
        if self.activation_type == "relu":
            self.fn = F.relu
        elif self.activation_type == "gelu":
            self.fn = F.gelu
        elif self.activation_type == "swiglu":
            # Alias for silu * x
            self.fn = F.silu
        else:
            raise OLMoConfigurationError(f"Unknown activation type: {activation_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == "swiglu":
            # SwiGLU is defined as Silu(xW + b) * (xV + c)
            # This is a bit of a hack, but it works for now.
            # We assume that the input x has already been split into two parts.
            # See modeling_olmo.py for more details.
            # TODO: make this more general.
            x, gate = x.chunk(2, dim=-1)
            return x * self.fn(gate)
        else:
            return self.fn(x)


class LayerNormBase(nn.Module):
    """
    Base class for layer normalization layers.
    """

    def __init__(self, normalized_shape: Union[int, list, torch.Size], eps: float, elementwise_affine: bool):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*self.normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LowPrecisionLayerNorm(LayerNormBase):
    """
    Layer normalization layer that performs the mean and variance calculations in float32
    but the affine transformation in the input dtype.
    This is similar to `torch.nn.LayerNorm` but with the option to disable the affine transformation
    and to control the epsilon value.
    """

    def __init__(
        self,
        normalized_shape: Union[int, list, torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module_device = x.device
        downcast_x = x.float()
        downcast_weight = self.weight.float() if self.weight is not None else None
        downcast_bias = self.bias.float() if self.bias is not None else None
        with torch.autocast(enabled=False, device_type=module_device.type):
            out = F.layer_norm(
                downcast_x, self.normalized_shape, downcast_weight, downcast_bias, self.eps
            )
        return out.to(x.dtype)


class RMSNorm(LayerNormBase):
    """
    Root mean square layer normalization.
    This is similar to `torch.nn.LayerNorm` but with the option to disable the affine transformation
    and to control the epsilon value.
    It also uses a different normalization formula: `x / sqrt(mean(x^2) + eps) * weight + bias`.
    """

    def __init__(
        self,
        normalized_shape: Union[int, list, torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: this is not numerically stable.
        # See https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/misc/rmsnorm.py
        # for a more stable implementation.
        norm_x = x.norm(2, dim=-1, keepdim=True)
        x_float = x.float()
        rms_x = norm_x * norm_x * (1 / x.shape[-1])  # type: ignore
        x_normed = x_float / torch.sqrt(rms_x + self.eps)
        if self.elementwise_affine:
            x_normed = x_normed * self.weight.float() + self.bias.float()
        return x_normed.to(x.dtype)


# Distributed utilities.
# These are just wrappers around `torch.distributed` functions.
# We use these wrappers to make it easier to mock these functions in tests.


def get_global_rank() -> int:
    if is_distributed():
        return torch.distributed.get_rank()
    else:
        return 0


def get_world_size() -> int:
    if is_distributed():
        return torch.distributed.get_world_size()
    else:
        return 1


def barrier() -> None:
    if is_distributed():
        torch.distributed.barrier()


def all_reduce(tensor: torch.Tensor, op: Any = None, group: Any = None, async_op: bool = False) -> None:
    if is_distributed():
        torch.distributed.all_reduce(tensor, op=op, group=group, async_op=async_op)


def all_gather(
    tensor_list: list[torch.Tensor], tensor: torch.Tensor, group: Any = None, async_op: bool = False
) -> None:
    if is_distributed():
        torch.distributed.all_gather(tensor_list, tensor, group=group, async_op=async_op)


def broadcast(tensor: torch.Tensor, src: int, group: Any = None, async_op: bool = False) -> None:
    if is_distributed():
        torch.distributed.broadcast(tensor, src, group=group, async_op=async_op)


def reduce_sum(tensor: torch.Tensor, destination: int, group: Any = None) -> None:
    if is_distributed():
        torch.distributed.reduce(tensor, destination, op=torch.distributed.ReduceOp.SUM, group=group)


def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def is_rank0() -> bool:
    return get_global_rank() == 0


# FSDP utilities.
# These are just wrappers around FSDP functions and enums.
# We use these wrappers to make it easier to mock these functions in tests,
# and to avoid importing FSDP unless it's actually needed.


def get_fsdp_wrap_module_name() -> str:
    # TODO: make this configurable.
    return "OLMoBlock"


def get_fsdp_wrap_module_type() -> type:
    from olmo.model import OLMoBlock  # type: ignore

    return OLMoBlock


def get_fsdp_device_mesh() -> Any:
    # TODO: make this configurable.
    return None


def get_fsdp_world_size(device_mesh: Optional[Any] = None) -> int:
    if device_mesh is not None:
        return device_mesh.size()  # type: ignore
    else:
        return get_world_size()


def get_fsdp_rank(device_mesh: Optional[Any] = None) -> int:
    if device_mesh is not None:
        # TODO: this might not be right for all device meshes.
        return device_mesh.get_coordinate()[0]  # type: ignore
    else:
        return get_global_rank()


def get_fsdp_strategy(name: str) -> Any:
    from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy

    if name == "full_shard":
        return ShardingStrategy.FULL_SHARD
    elif name == "shard_grad_op":
        return ShardingStrategy.SHARD_GRAD_OP
    elif name == "no_shard":
        return ShardingStrategy.NO_SHARD
    elif name == "hybrid_full_shard":
        return ShardingStrategy.HYBRID_SHARD
    elif name == "hybrid_shard_grad_op":
        # TODO: Not sure what the right way to specify this is.
        # For now, we'll just use HYBRID_SHARD.
        return ShardingStrategy.HYBRID_SHARD
    else:
        raise OLMoConfigurationError(f"Unknown FSDP strategy: {name}")


def get_fsdp_auto_wrap_policy(module_name: Optional[str] = None) -> Optional[Callable]:
    from functools import partial

    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    from olmo.model import OLMoBlock  # type: ignore

    if module_name is None:
        module_name = get_fsdp_wrap_module_name()
    if module_name == "OLMoBlock":
        return partial(transformer_auto_wrap_policy, transformer_layer_cls={OLMoBlock})
    else:
        # TODO: add support for other policies.
        raise OLMoConfigurationError(f"Unknown FSDP auto wrap policy: {module_name}")


def get_fsdp_cpu_offload(name: str) -> Any:
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

    if name == "none":
        return CPUOffload(offload_params=False)
    elif name == "sharded":
        return CPUOffload(offload_params=True)
    else:
        raise OLMoConfigurationError(f"Unknown FSDP CPU offload type: {name}")


def get_fsdp_mixed_precision_policy(config: dict[str, Any], device_mesh: Optional[Any] = None) -> Any:
    from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision

    # TODO: make this configurable.
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )


def get_fsdp_backward_prefetch_policy(name: str) -> Any:
    from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch

    if name == "backward_pre":
        return BackwardPrefetch.BACKWARD_PRE
    elif name == "backward_post":
        return BackwardPrefetch.BACKWARD_POST
    else:
        # TODO: add support for other policies.
        raise OLMoConfigurationError(f"Unknown FSDP backward prefetch policy: {name}")


def get_fsdp_sharding_strategy(name: str) -> Any:
    from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy

    if name == "full_shard":
        return ShardingStrategy.FULL_SHARD
    elif name == "shard_grad_op":
        return ShardingStrategy.SHARD_GRAD_OP
    elif name == "no_shard":
        return ShardingStrategy.NO_SHARD
    elif name == "hybrid_full_shard":
        return ShardingStrategy.HYBRID_SHARD
    elif name == "hybrid_shard_grad_op":
        # TODO: Not sure what the right way to specify this is.
        # For now, we'll just use HYBRID_SHARD.
        return ShardingStrategy.HYBRID_SHARD
    else:
        raise OLMoConfigurationError(f"Unknown FSDP sharding strategy: {name}")


def get_fsdp_limit_all_gathers(value: bool) -> bool:
    return value


def get_fsdp_use_orig_params(value: bool) -> bool:
    return value


def fsdp_model_state_dict_rank0(model: nn.Module, rank0_only: bool = True) -> dict[str, Any]:
    """
    Get the model state dict on rank 0 only.
    """
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    if rank0_only and not is_rank0():
        # Other ranks don't need to participate in this.
        return {}

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        return model.state_dict()


def fsdp_full_optim_state_dict_rank0(
    model: nn.Module, optim: torch.optim.Optimizer, rank0_only: bool = True
) -> dict[str, Any]:
    """
    Get the full optimizer state dict on rank 0 only.
    """
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    if rank0_only and not is_rank0():
        # Other ranks don't need to participate in this.
        return {}

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        return FSDP.full_optim_state_dict(model, optim)


class TorchCompileConfig:
    def __init__(self, mode: Optional[str] = None, options: Optional[dict[str, Any]] = None):
        self.mode = mode
        self.options = options


def maybe_compile(config: TorchCompileConfig, model: nn.Module) -> nn.Module:
    if config.mode is not None:
        return torch.compile(model, mode=config.mode, options=config.options)  # type: ignore
    else:
        return model
