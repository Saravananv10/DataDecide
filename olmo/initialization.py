import math
import warnings
from typing import Callable, Optional, Union

import torch
from torch import nn

from .config import ModelConfig

ModuleType = Union[nn.Linear, nn.Embedding, nn.LayerNorm]


def initialize_parameters(module: nn.Module, init_fn_name: str = "mitchell"):
    """
    Initialize the parameters of a module.

    :param module: The module to initialize.
    :param init_fn_name: The name of the initialization function to use.
        Choices: "mitchell", "kaiming_normal", "small_init".
    """
    init_fn = get_init_fn(init_fn_name)
    # Use a custom initialization function for LayerNorm and Embedding layers.
    # For other layers, just use the default initialization function.
    # We also need to handle the case where the module has submodules.
    # In this case, we recursively call this function on the submodules.
    # We also need to handle the case where the module has parameters directly.
    # In this case, we just apply the initialization function to the parameters.
    # This is not ideal, but it's the best we can do for now.
    # TODO: make this more robust.
    if isinstance(module, (nn.Linear, nn.Embedding)):
        init_fn(module)
    elif isinstance(module, (nn.LayerNorm, nn.modules.normalization.LayerNorm)):
        # LayerNorm has a very specific initialization scheme.
        # We initialize the weights to 1 and the biases to 0.
        # This is because LayerNorm is usually used as a normalization layer,
        # and we don't want to change the mean and variance of the input.
        # We also don't want to scale the output.
        # This is why we initialize the weights to 1 and the biases to 0.
        # This is also what fairseq does.
        # See https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/layer_norm.py#L22
        if module.elementwise_affine:
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
    elif isinstance(module, nn.ModuleList):
        for m in module:
            initialize_parameters(m, init_fn_name)
    elif isinstance(module, nn.ModuleDict):
        for m in module.values():
            initialize_parameters(m, init_fn_name)
    elif len(list(module.children())) > 0:
        # This module has submodules, so we recursively call this function on them.
        for m in module.children():
            initialize_parameters(m, init_fn_name)
    elif len(list(module.parameters(recurse=False))) > 0:
        # This module has parameters directly, so we just apply the initialization function to them.
        # This is not ideal, but it's the best we can do for now.
        # TODO: make this more robust.
        for p in module.parameters(recurse=False):
            if p.requires_grad:
                # We only initialize parameters that require gradients.
                # This is because some parameters (e.g. buffers) don't require gradients.
                # We also don't want to initialize parameters that are already initialized.
                # This is why we only initialize parameters that require gradients.
                # This is also what fairseq does.
                # See https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/multihead_attention.py#L175
                # We also need to handle the case where the parameter is a bias term.
                # In this case, we initialize it to 0.
                # This is because bias terms are usually initialized to 0.
                # This is also what fairseq does.
                # See https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/multihead_attention.py#L176
                if p.dim() > 1:
                    init_fn(p)
                else:
                    nn.init.zeros_(p)
    else:
        # This module has no parameters and no submodules, so we do nothing.
        pass


def get_init_fn(init_fn_name: str) -> Callable[[ModuleType], None]:
    """
    Get an initialization function by name.

    :param init_fn_name: The name of the initialization function to use.
        Choices: "mitchell", "kaiming_normal", "small_init".
    """
    if init_fn_name == "mitchell":
        return init_weights_mitchell
    elif init_fn_name == "kaiming_normal":
        return init_weights_kaiming_normal
    elif init_fn_name == "small_init":
        return init_weights_small_init
    else:
        raise ValueError(f"Unknown initialization function: {init_fn_name}")


def init_weights_mitchell(module: ModuleType, std: float = 0.02):
    """
    Initialize weights according to the GPT-2 paper.
    This is also the default initialization for HuggingFace GPT-2.
    See https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L302
    This is also the default initialization for OLMo.
    See https://github.com/allenai/OLMo/blob/main/olmo/model.py#L302
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    else:
        warnings.warn(
            f"Skipping initialization for {module}, not recognized as a Linear, Embedding, or LayerNorm"
        )


def init_weights_kaiming_normal(module: ModuleType, nonlinearity: str = "relu"):
    """
    Initialize weights using Kaiming normal initialization.
    This is the default initialization for PyTorch Linear layers.
    See https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.kaiming_normal_(module.weight, nonlinearity=nonlinearity)
        if isinstance(module, nn.Linear) and module.bias is not None:
            # Initialize bias to zero.
            # This is what PyTorch does by default.
            # See https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
            # We use a fan-in that corresponds to the Kaiming normal initialization.
            # This is what fairseq does.
            # See https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/linear.py#L19
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, nn.LayerNorm):
        # LayerNorm has a very specific initialization scheme.
        # We initialize the weights to 1 and the biases to 0.
        # This is because LayerNorm is usually used as a normalization layer,
        # and we don't want to change the mean and variance of the input.
        # We also don't want to scale the output.
        # This is why we initialize the weights to 1 and the biases to 0.
        # This is also what fairseq does.
        # See https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/layer_norm.py#L22
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()
    else:
        warnings.warn(
            f"Skipping initialization for {module}, not recognized as a Linear, Embedding, or LayerNorm"
        )


def init_weights_small_init(module: ModuleType, dim: Optional[int] = None):
    """
    Initialize weights using a small normal distribution.
    This is used in some of the earlier GPT models.
    See https://github.com/openai/gpt-2/blob/master/src/model.py#L194
    This is also used in some of the earlier OLMo models.
    See https://github.com/allenai/OLMo/blob/main/olmo/model.py#L302
    """
    if dim is None:
        if not hasattr(module, "weight") or module.weight is None:
            warnings.warn(f"Skipping initialization for {module}, it has no attribute 'weight'")
            return
        dim = module.weight.size(0)  # type: ignore
    std = math.sqrt(2 / (5 * dim))  # TODO: make this configurable
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        # LayerNorm has a very specific initialization scheme.
        # We initialize the weights to 1 and the biases to 0.
        # This is because LayerNorm is usually used as a normalization layer,
        # and we don't want to change the mean and variance of the input.
        # We also don't want to scale the output.
        # This is why we initialize the weights to 1 and the biases to 0.
        # This is also what fairseq does.
        # See https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/layer_norm.py#L22
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()
    else:
        warnings.warn(
            f"Skipping initialization for {module}, not recognized as a Linear, Embedding, or LayerNorm"
        )
