"""Configuration for OLMo.

This is a schema for OLMo model and training configurations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .aliases import PathOrStr
from .exceptions import OLMoConfigurationError

__all__ = [
    "ActivationType",
    "ActivationFunction",
    "BlockType",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "EvaluatorConfig",
    "TokenizerConfig",
    "DataConfig",
    "TrainConfig",
]

ActivationType = "relu"
"""
The non-linearity, chosen from `torch.nn.functional`
"""

ActivationFunction = "F.relu"
"""
The activation function to use.
"""

BlockType = "sequential"
"""
The type of block to use.
"""


@dataclass
class ModelConfig:
    """
    OLMo model configuration.
    """

    # Name of the model (for logging, etc.)
    model_name: str = "olmo"

    # Architecture arguments.
    d_model: int = 768
    """
    The hidden size of the model.
    """

    n_layers: int = 12
    """
    The number of layers in the model.
    """

    n_heads: int = 12
    """
    The number of attention heads.
    """

    # The following two parameters are used to compute `n_kv_heads`.
    # We provide `n_kv_heads` directly as a convenience for cases where
    # we want to specify it directly.
    # If `n_kv_heads` is not None, then `attention_layer_layout` and
    # `attention_layers_sharing_groups` will be ignored.
    # Otherwise, `n_kv_heads` is computed from `attention_layer_layout` and
    # `attention_layers_sharing_groups`.
    # `n_kv_heads` must be a divisor of `n_heads`.
    # The default behavior is MHA, so `n_kv_heads` is equal to `n_heads`.
    n_kv_heads: Optional[int] = None
    """
    Optional[int]: The number of key/value heads for Grouped Query Attention.
    If specified, this overrides `attention_layer_layout` and `attention_layers_sharing_groups`.
    Default: `None` (equivalent to MHA, so `n_kv_heads` = `n_heads`).
    """

    # The following two parameters are used to compute `n_kv_heads`.
    # `attention_layer_layout` specifies the number of attention heads for each layer.
    # `attention_layers_sharing_groups` specifies how many consecutive layers share the same
    # key/value projections.
    # For example, if `n_heads` = 12, `attention_layer_layout` = (4, 8, 12), and
    # `attention_layers_sharing_groups` = (4, 4, 4), then
    # layers 0-3 will have 4 query heads and 4 key/value heads,
    # layers 4-7 will have 8 query heads and 8 key/value heads,
    # layers 8-11 will have 12 query heads and 12 key/value heads.
    # If `attention_layer_layout` is a tuple, then `attention_layers_sharing_groups` must also be a tuple
    # of the same length, and the sum of `attention_layers_sharing_groups` must be equal to `n_layers`.
    # If `attention_layer_layout` is an int, then `attention_layers_sharing_groups` must also be an int,
    # and `attention_layers_sharing_groups` must be equal to `n_layers`.
    # The default behavior is MHA, so `attention_layer_layout` = `n_heads` and
    # `attention_layers_sharing_groups` = `n_layers`.
    attention_layer_layout: Union[Tuple[int, ...], int] = -1  # Placeholder for default
    """
    Union[Tuple[int, ...], int]: The number of attention heads for each layer.
    Default: -1 (placeholder for MHA, actual default set in `__post_init__`).
    """

    attention_layers_sharing_groups: Union[Tuple[int, ...], int] = -1  # Placeholder for default
    """
    Union[Tuple[int, ...], int]: How many consecutive layers share the same key/value projections.
    Default: -1 (placeholder for MHA, actual default set in `__post_init__`).
    """

    mlp_hidden_size: Optional[int] = None
    """
    The hidden size of the MLP. If ``None``, it will be set to ``4 * d_model``.
    """

    mlp_ratio: Optional[int] = None
    """
    The ratio of the MLP hidden size to the model dimension.
    If ``mlp_hidden_size`` is specified, this is ignored. Otherwise, if this is specified,
    the MLP hidden size will be ``mlp_ratio * d_model``.
    If neither ``mlp_hidden_size`` nor ``mlp_ratio`` are specified, ``mlp_ratio`` will be set to 4.
    """

    activation_type: ActivationType = "swiglu"
    """
    The activation function to use.
    """

    block_type: BlockType = "sequential"
    """
    The type of block to use.
    """

    block_group_size: int = 1
    """
    The number of blocks to group together. This is only used if ``block_type`` is ``grouped``.
    """

    alibi: bool = False
    """
    Whether to use ALiBi positional embeddings. This is only used if ``rope`` is ``False``.
    If ``True``, ``rope`` must be ``False``.
    """

    alibi_bias_max: float = 8.0
    """
    The maximum bias value for ALiBi.
    """

    rope: bool = True
    """
    Whether to use rotary positional embeddings. This is only used if ``alibi`` is ``False``.
    If ``True``, ``alibi`` must be ``False``.
    """

    rope_full_precision: bool = True
    """
    Whether to use full precision for rotary embeddings.
    """

    # Tokenizer arguments.
    embedding_size: Optional[int] = 50257  # TODO: remove this, should be part of tokenizer_config
    """
    The size of the token embedding.
    The hidden size of the model must be divisible by the embedding size.
    """

    # Initialization arguments.
    init_fn: str = "mitchell"
    """
    The initialization function to use.
    """

    init_std: float = 0.02  # TODO: remove this, should be part of init_fn_args
    """
    The standard deviation for initialization.
    """

    # Other arguments.
    eos_token_id: int = 50256
    """
    The ID of the end-of-sentence token.
    """

    pad_token_id: int = 50256
    """
    The ID of the padding token.
    """

    bos_token_id: int = 50256  # TODO: remove this, should be part of tokenizer_config
    """
    The ID of the beginning-of-sentence token.
    """

    # Generation arguments
    scale_logits: bool = False
    """
    Whether to scale the logits by ``1 / sqrt(d_model)``.
    """

    # Attention arguments
    attention_dropout: float = 0.1
    """
    The dropout probability for the attention layers.
    """

    multi_query_attention: bool = False  # TODO: remove this, should be specified with `n_kv_heads`
    """
    Whether to use multi-query attention.
    If ``True``, the number of key/value heads will be 1. Otherwise, it will be ``n_heads``.
    This is only used if ``n_kv_heads`` is not specified.
    """

    # Layer norm arguments
    layer_norm_type: str = "default"
    """
    The type of layer norm to use. Options: "default", "low_precision", "rms".
    "default" is `torch.nn.LayerNorm`.
    "low_precision" is `olmo.torch_util.LowPrecisionLayerNorm`.
    "rms" is `olmo.torch_util.RMSNorm`.
    """

    layer_norm_with_affine: bool = True
    """
    Whether to include the affine transformation in the layer norm.
    """

    # Embedding arguments
    embedding_dropout: float = 0.1
    """
    The dropout probability for the embedding layer.
    """

    # Residual arguments
    residual_dropout: float = 0.1
    """
    The dropout probability for the residual connections.
    """

    # Flash Attention arguments.
    flash_attention: bool = False  # TODO: remove this, should be automatically determined
    """
    Whether to use Flash Attention.
    """

    attention_softmax_fp32: bool = True  # TODO: remove this, should be part of flash_attention_args
    """
    Whether to use FP32 for the attention softmax.
    """

    # Miscellaneous arguments.
    max_sequence_length: int = 1024
    """
    The maximum sequence length.
    """

    vocab_size: int = 50257  # TODO: remove this, should be part of tokenizer_config
    """
    The size of the vocabulary.
    """

    include_bias: bool = True
    """
    Whether to include bias terms in linear layers and layer norms.
    """

    bias_for_layer_norm: Optional[bool] = None
    """
    Whether to include bias terms in layer norms. If ``None``, this will be set to ``include_bias``.
    """

    weight_tying: bool = True
    """
    Whether to tie the input and output embeddings.
    """

    precision: str = "fp32"
    """
    The precision to use for training. Options: "fp32", "bf16", "fp16".
    """

    # Low precision layer norm arguments (only used if `layer_norm_type` is "low_precision")
    low_precision_layer_norm_epsilon: float = 1e-5
    """
    The epsilon value for low precision layer norm.
    """

    # RMS norm arguments (only used if `layer_norm_type` is "rms")
    rms_norm_epsilon: float = 1e-5
    """
    The epsilon value for RMS norm.
    """

    # For compatibility with older configs.
    # TODO: remove these later.
    residual_dropout_prob: Optional[float] = None
    embedding_dropout_prob: Optional[float] = None
    attention_dropout_prob: Optional[float] = None

    def __post_init__(self):
        # Handle deprecated fields.
        # TODO: remove these later.
        if self.residual_dropout_prob is not None:
            self.residual_dropout = self.residual_dropout_prob
        if self.embedding_dropout_prob is not None:
            self.embedding_dropout = self.embedding_dropout_prob
        if self.attention_dropout_prob is not None:
            self.attention_dropout = self.attention_dropout_prob

        # Validate arguments.
        if self.alibi and self.rope:
            raise OLMoConfigurationError("Cannot use both ALiBi and rope embeddings")
        if self.mlp_hidden_size is not None and self.mlp_ratio is not None:
            raise OLMoConfigurationError("Cannot specify both mlp_hidden_size and mlp_ratio")
        if self.mlp_hidden_size is None and self.mlp_ratio is None:
            self.mlp_ratio = 4
        if self.mlp_ratio is not None:
            self.mlp_hidden_size = int(self.d_model * self.mlp_ratio)
        assert self.mlp_hidden_size is not None

        if self.n_kv_heads is not None:
            if self.n_heads % self.n_kv_heads != 0:
                raise OLMoConfigurationError(
                    f"`n_heads` ({self.n_heads}) must be divisible by `n_kv_heads` ({self.n_kv_heads})"
                )
            if self.multi_query_attention:
                # TODO: remove `multi_query_attention` eventually.
                raise OLMoConfigurationError(
                    "`multi_query_attention` is deprecated. Use `n_kv_heads=1` instead."
                )
        elif self.multi_query_attention:
            # TODO: remove `multi_query_attention` eventually.
            self.n_kv_heads = 1
        else:
            self.n_kv_heads = self.n_heads

        if self.attention_layer_layout == -1:
            self.attention_layer_layout = self.n_heads
        if self.attention_layers_sharing_groups == -1:
            self.attention_layers_sharing_groups = self.n_layers

        if isinstance(self.attention_layer_layout, int):
            self.attention_layer_layout = (self.attention_layer_layout,) * self.n_layers
        if isinstance(self.attention_layers_sharing_groups, int):
            self.attention_layers_sharing_groups = (self.attention_layers_sharing_groups,) * self.n_layers

        if len(self.attention_layer_layout) != self.n_layers:
            raise OLMoConfigurationError(
                f"Length of `attention_layer_layout` ({len(self.attention_layer_layout)}) "
                f"must be equal to `n_layers` ({self.n_layers})"
            )
        if len(self.attention_layers_sharing_groups) != self.n_layers:
            raise OLMoConfigurationError(
                f"Length of `attention_layers_sharing_groups` ({len(self.attention_layers_sharing_groups)}) "
                f"must be equal to `n_layers` ({self.n_layers})"
            )

        if self.bias_for_layer_norm is None:
            self.bias_for_layer_norm = self.include_bias


@dataclass
class OptimizerConfig:
    name: str = "AdamW"
    """Name of the optimizer class from ``torch.optim``."""

    learning_rate: float = 3e-4
    """The learning rate."""

    weight_decay: float = 0.1
    """The weight decay."""

    betas: Tuple[float, float] = field(default_factory=lambda: (0.9, 0.95))
    """Optimizer specific betas."""

    eps: float = 1e-8
    """Optimizer specific epsilon."""

    args: Dict[str, Any] = field(default_factory=dict)
    """Other arguments to pass to the optimizer. For example, ``{"fused": True}`` for AdamW."""


@dataclass
class SchedulerConfig:
    name: str = "linear_with_warmup"
    """The name of the scheduler to use."""

    t_warmup: Union[str, int] = "1000ba"
    """The number of warmup steps or a string like '1000ba' or '0.1ep'."""

    t_max: Union[str, int] = "100000ba"
    """The total number of training steps or a string like '100000ba' or '1ep'."""

    alpha_f: float = 0.1
    """The final learning rate as a fraction of the initial learning rate."""

    args: Dict[str, Any] = field(default_factory=dict)
    """Other arguments to pass to the scheduler."""


@dataclass
class EvaluatorConfig:
    label: str
    """The label for this evaluator. This is used for logging."""

    type: str
    """The type of evaluator to use."""

    data_path: PathOrStr
    """The path to the data for this evaluator."""

    # Optional arguments for the data loader.
    # These are passed directly to `torch.utils.data.DataLoader`.
    data_loader_args: Dict[str, Any] = field(default_factory=dict)

    # Optional arguments for the evaluator itself.
    # These are passed directly to the evaluator constructor.
    evaluator_args: Dict[str, Any] = field(default_factory=dict)

    # Optional subset of data to use for evaluation.
    subset_num_batches: Optional[int] = None
    """The number of batches to use from the evaluation data. If ``None``, all data is used."""


@dataclass
class TokenizerConfig:
    identifier: str
    """The tokenizer identifier. This is usually the path to the tokenizer file."""

    # Optional arguments for the tokenizer.
    # These are passed directly to `transformers.AutoTokenizer.from_pretrained`.
    tokenizer_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    paths: List[PathOrStr]
    """A list of paths to the data files or directories."""

    # Optional arguments for the data loader.
    # These are passed directly to `torch.utils.data.DataLoader`.
    # If `None`, default arguments will be used.
    # Example: `{"num_workers": 8, "pin_memory": True, "drop_last": True}`
    data_loader_args: Optional[Dict[str, Any]] = None

    # Optional arguments for the dataset itself.
    # These are passed directly to the dataset constructor.
    dataset_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    """
    OLMo training configuration.
    """

    # Required arguments.
    model: ModelConfig
    """The model configuration."""

    optimizer: OptimizerConfig
    """The optimizer configuration."""

    scheduler: SchedulerConfig
    """The scheduler configuration."""

    data: DataConfig
    """The data configuration."""

    # Optional arguments.
    seed: int = 6198
    """The random seed."""

    run_name: Optional[str] = None
    """The name of the run. If ``None``, a name will be generated automatically."""

    # Checkpointing arguments.
    save_folder: Optional[PathOrStr] = None
    """The root folder where checkpoints will be saved."""

    save_interval: int = 1000
    """The interval (in batches) at which to save checkpoints."""

    save_num_checkpoints_to_keep: int = -1
    """The number of checkpoints to keep. If -1, all checkpoints are kept."""

    save_overwrite: bool = False
    """Whether to overwrite existing checkpoints."""

    load_path: Optional[PathOrStr] = None
    """The path to a checkpoint to load from."""

    load_weights_only: bool = False
    """Whether to only load the model weights from the checkpoint."""

    load_strict_matching: bool = True
    """Whether to use strict matching when loading the model weights."""

    # Distributed training arguments.
    device_train_batch_size: Optional[int] = None
    """The batch size for training on each device."""

    device_train_grad_accum: Optional[int] = None
    """The number of gradient accumulation steps to use on each device for training."""

    # Precision arguments.
    precision: Optional[str] = None  # TODO: remove this, should be part of model_config
    """The precision to use for training. Options: "fp32", "bf16", "fp16"."""

    # Evaluation arguments.
    eval_interval: int = 1000
    """The interval (in batches) at which to run evaluation."""

    evaluators: List[EvaluatorConfig] = field(default_factory=list)
    """A list of evaluators to run."""

    device_eval_batch_size: Optional[int] = None  # TODO: remove this, should be part of data_loader_args
    """The batch size for evaluation on each device."""

    # Logging arguments.
    wandb: Optional[Dict[str, Any]] = None
    """
    Optional arguments for Weights & Biases.
    If ``None``, W&B will not be used. Otherwise, this should be a dictionary of arguments
    to pass to ``wandb.init()``. For example, ``{"project": "my-project"}``.
    """

    console_log_interval: int = 10
    """The interval (in batches) at which to log to the console."""

    # Other arguments.
    max_duration: Union[str, int] = "100000ba"
    """The maximum duration of training, e.g. '100000ba' or '1ep'."""

    max_sequence_length: Optional[int] = None  # TODO: remove this, should be part of model_config
    """The maximum sequence length."""

    # FSDP arguments.
    fsdp: Optional[Dict[str, Any]] = None
    """
    Optional arguments for FullyShardedDataParallel.
    If ``None``, FSDP will not be used. Otherwise, this should be a dictionary of arguments
    to pass to ``torch.distributed.fsdp.FullyShardedDataParallel``.
    """

    # For compatibility with older configs.
    # TODO: remove these later.
    batch_size: Optional[int] = None
    grad_accum_steps: Optional[int] = None

    def __post_init__(self):
        # Handle deprecated fields.
        # TODO: remove these later.
        if self.batch_size is not None:
            self.device_train_batch_size = self.batch_size
        if self.grad_accum_steps is not None:
            self.device_train_grad_accum = self.grad_accum_steps

        # Validate arguments.
        if self.precision is not None:
            self.model.precision = self.precision
        if self.max_sequence_length is not None:
            self.model.max_sequence_length = self.max_sequence_length
        if self.model.max_sequence_length <= 0:
            raise OLMoConfigurationError("max_sequence_length must be positive")
        if self.device_train_batch_size is None or self.device_train_batch_size <= 0:
            raise OLMoConfigurationError("device_train_batch_size must be positive")
        if self.device_train_grad_accum is None or self.device_train_grad_accum <= 0:
            raise OLMoConfigurationError("device_train_grad_accum must be positive")

        # Set defaults.
        if self.run_name is None:
            # TODO: come up with a better way to name runs.
            self.run_name = self.model.model_name

        if self.device_eval_batch_size is None:
            self.device_eval_batch_size = self.device_train_batch_size

        if self.fsdp is not None and self.fsdp.get("wrapping_strategy") == "by_block_and_size":
            # These are required for this wrapping strategy.
            if "block_size_mb" not in self.fsdp:
                self.fsdp["block_size_mb"] = 20
            if "min_wrapped_params" not in self.fsdp:
                self.fsdp["min_wrapped_params"] = 2**18

    @property
    def global_train_batch_size(self) -> int:
        # TODO: this needs to be updated for FSDP.
        from torch.distributed import get_world_size

        if self.device_train_batch_size is None or self.device_train_grad_accum is None:
            # Should be caught by __post_init__ but mypy doesn't know that.
            raise OLMoConfigurationError(
                "device_train_batch_size and device_train_grad_accum must be set"
            )
        return self.device_train_batch_size * self.device_train_grad_accum * get_world_size()

    @property
    def global_eval_batch_size(self) -> int:
        # TODO: this needs to be updated for FSDP.
        from torch.distributed import get_world_size

        if self.device_eval_batch_size is None:
            # Should be caught by __post_init__ but mypy doesn't know that.
            raise OLMoConfigurationError("device_eval_batch_size must be set")
        return self.device_eval_batch_size * get_world_size()

    def save_fsdp_model_config(self, model_config: ModelConfig, save_dir: PathOrStr):
        """
        Save the model config to a file in the save directory.
        """
        # TODO: implement this.
        pass

    def load_fsdp_model_config(self, save_dir: PathOrStr) -> ModelConfig:
        """
        Load the model config from a file in the save directory.
        """
        # TODO: implement this.
        raise NotImplementedError

    def save_fsdp_train_config(self, train_config: "TrainConfig", save_dir: PathOrStr):
        """
        Save the train config to a file in the save directory.
        """
        # TODO: implement this.
        pass

    def load_fsdp_train_config(self, save_dir: PathOrStr) -> "TrainConfig":
        """
        Load the train config from a file in the save directory.
        """
        # TODO: implement this.
        raise NotImplementedError
