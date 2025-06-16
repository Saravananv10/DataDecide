# Standard library imports
import os  # For operating system interactions like file paths and environment variables
import math  # For mathematical functions like cosine in LR scheduler
import time  # For timing operations (e.g., calculating step duration)
import inspect  # For inspecting function signatures (e.g., AdamW fused optimizer)

# Third-party library imports
import torch  # Main PyTorch library
import torch.nn as nn  # Neural network modules from PyTorch
from torch.nn import functional as F  # Common neural network functions (e.g., softmax)
from torch.distributed import init_process_group, destroy_process_group  # For DDP setup and cleanup
from torch.nn.parallel import DistributedDataParallel as DDP  # DDP wrapper for models
import torch.distributed as dist # For distributed operations like all_reduce

import tiktoken  # For tokenization (used here for vocab size and sample generation)

# OLMo library imports (from Hugging Face / AllenAI)
from hf_olmo import OLMoConfig, OLMoForCausalLM  # Core OLMo model configuration and causal language model class

# --- User-provided module imports ---
# These modules are expected to be in the same directory as this script or in the Python path.

# `evaluator.py`: Contains custom evaluation logic.
# It should define a function `run_evaluations` that takes the model, config, device, DDP info,
# log file path, and current step, and returns a dictionary of evaluation metrics.
try:
    import evaluator  # Attempt to import the user's custom evaluator module
except ImportError:
    print("Warning: 'evaluator.py' not found. Custom evaluation calls will fail.")
    evaluator = None  # Set to None if not found, allowing the script to run without custom evals

# `DataLoaderLite`: A simple data loader for pre-tokenized data.
# This class is included directly in this script for simplicity but could be in a separate file.
# It reads data shards (.npy files) and serves batches for training and validation.
class DataLoaderLite:
    """
    Minimalistic data loader for pre-tokenized data stored in .npy shards.
    Handles DDP by loading a subset of data corresponding to the process rank.
    """
    def __init__(self, B, T, process_rank, num_processes, split, data_root="fineweb_custom"):
        """
        Initializes DataLoaderLite.
        Args:
            B (int): Micro-batch size (number of sequences per device per step).
            T (int): Sequence length.
            process_rank (int): Rank of the current process in DDP.
            num_processes (int): Total number of DDP processes.
            split (str): Data split, e.g., "train" or "val".
            data_root (str): Root directory containing data shards.
        """
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.data_root = data_root  # Directory where data shards are stored

        # Load shard filenames for the specified split
        shards = os.listdir(self.data_root)
        shards = [s for s in shards if self.split in s and s.endswith('.npy')] # Ensure only .npy files for the split
        shards = sorted(shards)
        shards = [os.path.join(self.data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No .npy shards found for split '{split}' in '{self.data_root}'"

        self.reset()

    def reset(self):
        """Resets the data loader to the beginning of the shards."""
        self.current_shard = 0
        self.tokens = self._load_tokens(self.shards[self.current_shard])
        # Set current position based on DDP rank to ensure each process gets different data
        self.current_position = self.B * self.T * self.process_rank

    def _load_tokens(self, filename):
        """Loads tokens from a .npy file and converts them to a PyTorch tensor."""
        import numpy as np  # Local import to keep numpy dependency contained here
        npt = np.load(filename)
        npt = npt.astype(np.int32)  # Ensure tokens are integers
        ptt = torch.tensor(npt, dtype=torch.long)  # Convert to PyTorch long tensor
        return ptt

    def next_batch(self):
        """
        Returns the next batch of data (input x and target y).
        Handles advancing through shards and resetting when the end is reached.
        """
        B, T = self.B, self.T
        # Get a buffer of tokens for the current batch
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # Input tokens
        y = (buf[1:]).view(B, T)   # Target tokens (shifted by one)

        # Advance current position for the next batch
        self.current_position += B * T * self.num_processes

        # If current position exceeds available tokens in the shard, load the next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards) # Cycle through shards
            self.tokens = self._load_tokens(self.shards[self.current_shard])
            # Reset position based on rank for the new shard
            self.current_position = self.B * self.T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# Distributed Data Parallel (DDP) Setup
# This section configures DDP if multiple GPUs are used.
# It relies on environment variables set by `torchrun` or a similar launcher.

ddp = int(os.environ.get('RANK', -1)) != -1  # Check if RANK env var is set (indicates DDP)

if ddp:
    # DDP is enabled
    assert torch.cuda.is_available(), "DDP currently requires CUDA for NCCL backend"
    init_process_group(backend='nccl')  # Initialize DDP with NCCL backend (recommended for NVIDIA GPUs)
    ddp_rank = int(os.environ['RANK'])  # Global rank of the current process
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # Local rank on the current node
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # Total number of processes across all nodes
    device = f'cuda:{ddp_local_rank}'  # Assign specific GPU to this process
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)  # True if this is the master process (rank 0)
else:
    # DDP is not enabled (single GPU or CPU)
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # Determine device: CUDA, MPS (Apple Silicon), or CPU
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    if master_process: print(f"Using device: {device}")

# `device_type` is used for torch.autocast a few lines below.
device_type = "cuda" if device.startswith("cuda") else "cpu"

# Seed for reproducibility. Offset by DDP rank to ensure different initializations
# for different processes if that behavior is desired (e.g., for data shuffling).
torch.manual_seed(1337 + ddp_rank)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337 + ddp_rank)

# -----------------------------------------------------------------------------
# Model Configuration (OLMoConfig)
# Defines the architecture of the OLMo model to be trained.

# `enc` is a tiktoken encoder, used here to determine vocab_size from GPT-2 tokenizer.
# Replace "gpt2" with your target tokenizer if different.
# This is primarily for convenience; if you know your vocab_size, set it directly.
try:
    enc = tiktoken.get_encoding("gpt2")
    default_vocab_size = enc.n_words
except Exception as e:
    if master_process: print(f"Warning: tiktoken GPT-2 encoding not found ({e}). Using default vocab_size=50257.")
    default_vocab_size = 50257


# Parameters for the OLMo model. These define a model of ~20M parameters.
# Adjust these to scale the model up or down.
model_config_params = dict(
    d_model=192,                # Hidden dimension of the model.
    n_layers=16,                # Number of transformer layers.
    n_heads=8,                  # Number of attention heads in each layer.
    mlp_ratio=8,                # Ratio to determine the MLP (feed-forward) hidden size (d_ff = d_model * mlp_ratio).
    vocab_size=default_vocab_size, # Size of the vocabulary. For OLMo, this often matches `embedding_size`.
                                # GPT-2 tiktoken n_words is 50257. Some OLMo models use 50304.
    embedding_size=192,         # Dimension of token embeddings. Typically same as d_model.
                                # If `weight_tying` is True, OLMo may expect embedding_size == d_model.
    max_sequence_length=2048,   # Maximum sequence length the model can process. This must be >= T.
    weight_tying=True,          # Whether to tie input and output token embedding weights. Saves parameters.
    attention_dropout=0.0,      # Dropout rate for attention weights.
    residual_dropout=0.0,       # Dropout rate for residual connections.
    embedding_dropout=0.0,      # Dropout rate for token embeddings.
    dropout=0.0,                # General dropout rate (used in OLMo for FF layers, etc. if not specified otherwise).
    flash_attention=False,      # Whether to use FlashAttention (requires compatible hardware and installation). Set to True for potential speedups.
    # Other OLMoConfig specific parameters (e.g., `olmo_version`, `init_fn`, `init_std`)
    # will use their defaults from `hf_olmo.OLMoConfig` if not specified here.
)

if master_process:
    print(f"Initializing OLMo model with config: {model_config_params}")
    print(f"Note: vocab_size is {model_config_params['vocab_size']}. Adjust if using a specific OLMo tokenizer/vocab.")

# Create an OLMoConfig instance from the parameters
olmo_model_config = OLMoConfig(**model_config_params)

# -----------------------------------------------------------------------------
# Model Instantiation and Setup

if master_process: print("Creating OLMoForCausalLM model...")
# Instantiate the OLMo model with the defined configuration.
# This initializes the model with random weights.
model = OLMoForCausalLM(olmo_model_config)
model.to(device)  # Move the model to the determined device (GPU or CPU)

# Utility function to count trainable parameters in a model
def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

if master_process:
    total_params = count_parameters(model)
    print(f"Total trainable parameters (OLMo): {total_params:,}")
    # For reference, the AllenAI DataDecide 20M OLMo model has ~19.1M parameters
    # with d_model=192, n_layers=16, n_heads=8, mlp_ratio=8, vocab=50304.
    # This configuration should be very close to that.

# `use_compile` flag to enable `torch.compile` for potential speedups (PyTorch 2.0+).
# May not be compatible with all custom code or older PyTorch versions.
# The user's example script had issues with torch.compile and custom evaluations.
use_compile = False  # Default to False for broader compatibility initially.
if use_compile:
    if master_process:
        print("Compiling the model with torch.compile()...")
    try:
        model = torch.compile(model)
    except Exception as e:
        if master_process: print(f"torch.compile() failed: {e}. Proceeding without compilation.")
        use_compile = False # Fallback if compilation fails

# Wrap the model with DDP if DDP is enabled.
# This handles gradient synchronization across processes.
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # Get the underlying model instance (useful for saving, config access, etc.)

if master_process:
    print("Core script structure and model configuration complete.")
    print(f"Running on device: {device}")
    print(f"Master process: {master_process}, DDP rank: {ddp_rank}, DDP world size: {ddp_world_size}")

# -----------------------------------------------------------------------------
# Data Loading Setup

# Batching and data parameters
total_batch_size = 4096  # Effective total batch size in number of tokens across all devices and accumulation steps.
B = 2                    # Micro-batch size: Number of sequences processed per device in a single forward/backward pass.
T = 2048                 # Sequence length: Number of tokens in each sequence.
# Ensure T is not greater than the model's configured maximum sequence length.
assert T <= olmo_model_config.max_sequence_length, \
    f"Sequence length T ({T}) cannot exceed model's max_sequence_length ({olmo_model_config.max_sequence_length})"

# Calculate gradient accumulation steps.
# This allows achieving a larger effective batch size by accumulating gradients
# over multiple micro-batches before performing an optimizer step.
assert total_batch_size % (B * T * ddp_world_size) == 0, \
    "Total batch size must be divisible by (Micro B * Seq T * DDP World Size)"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"Total desired batch size (tokens): {total_batch_size}")
    print(f"Micro-batch size (B sequences): {B}, Sequence length (T tokens): {T}")
    print(f"Data root for DataLoaderLite: '{train_loader.data_root if 'train_loader' in locals() and hasattr(train_loader, 'data_root') else 'fineweb_custom (default)'}'") # Use actual if available
    print(f"Calculated gradient accumulation steps: {grad_accum_steps}")

# Instantiate DataLoaders for training and validation.
# These will use the DataLoaderLite class defined earlier.
# Ensure the data directory (e.g., 'fineweb_custom/') exists and is populated with
# pre-tokenized .npy shards (e.g., train_000.npy, val_000.npy).
try:
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")
    if master_process:
        print("Train and Val DataLoaderLite instances created.")
except FileNotFoundError as e:
    if master_process:
        print(f"Error initializing DataLoaderLite: {e}")
        print("Please ensure the data directory (e.g., 'fineweb_custom') exists relative to the script")
        print("and contains the necessary data shards (e.g., train_000.npy, val_000.npy).")
    # Exit if data loaders can't be initialized, as training cannot proceed.
    if ddp: destroy_process_group()
    exit(1)
except Exception as e:
    if master_process:
        print(f"An unexpected error occurred while initializing DataLoaderLite: {e}")
    if ddp: destroy_process_group()
    exit(1)

# Set matmul precision for `torch.float32` matrix multiplications.
# 'high' or 'highest' can improve performance on Ampere GPUs and newer by using TF32 cores.
torch.set_float32_matmul_precision('high')
if master_process:
    print("torch.set_float32_matmul_precision('high') set.")


# -----------------------------------------------------------------------------
# Optimizer and Learning Rate Scheduler Setup

# Optimizer hyperparameters
max_lr = 6e-4           # Maximum learning rate (also the initial LR for AdamW).
weight_decay = 0.1      # Weight decay for regularization (applied to non-bias/norm parameters).
adamw_beta1 = 0.9       # Beta1 parameter for AdamW optimizer.
adamw_beta2 = 0.95      # Beta2 parameter for AdamW optimizer.
adamw_eps = 1e-8        # Epsilon parameter for AdamW optimizer (for numerical stability).

# Learning rate scheduler hyperparameters
# `max_steps` defines the total number of training steps for the LR schedule.
# `warmup_steps` defines the number of initial steps for linear LR warmup.
# These should ideally be configurable (e.g., via command-line arguments or a config file).
max_steps = 19074       # Example total training steps (e.g., for ~1 epoch on a dataset).
warmup_steps = 950      # Example warmup steps (e.g., 5% of max_steps).
min_lr = max_lr * 0.1   # Minimum learning rate after cosine decay.

if master_process:
    print(f"Optimizer settings: max_lr={max_lr}, weight_decay={weight_decay}, AdamW betas=({adamw_beta1}, {adamw_beta2})")
    print(f"LR Scheduler settings: warmup_steps={warmup_steps}, max_steps={max_steps}, min_lr={min_lr}")

# Function to configure the optimizer with weight decay for specific parameter groups.
# Weight decay is typically applied to weight matrices (2D+ params) but not to biases or LayerNorm/RMSNorm parameters (1D params).
def configure_optimizers_refined(model_to_opt, wd_val, lr_val, betas_val, eps_val, device_type_val, print_names=False):
    """
    Sets up AdamW optimizer with selective weight decay.
    Args:
        model_to_opt: The model whose parameters will be optimized.
        wd_val (float): Weight decay value.
        lr_val (float): Learning rate.
        betas_val (tuple): AdamW beta1 and beta2 values.
        eps_val (float): AdamW epsilon value.
        device_type_val (str): "cuda" or "cpu", for fused optimizer option.
        print_names (bool): If True and master_process, prints names of decayed/non-decayed params.
    Returns:
        torch.optim.AdamW: The configured optimizer.
    """
    param_dict = {pn: p for pn, p in model_to_opt.named_parameters() if p.requires_grad}
    decay_params = []
    nodecay_params = []

    if master_process and print_names: print("--- Optimizer Parameter Groups ---")
    for pn, p in param_dict.items():
        # Apply weight decay to parameters that are 2D or higher (typically weight matrices)
        # and do not include ".bias" or normalization layer terms in their names.
        if p.dim() >= 2 and not (pn.endswith(".bias") or ".norm" in pn.lower() or ".ln" in pn.lower()):
            decay_params.append(p)
            if master_process and print_names: print(f"DECAY:    {pn} (shape: {p.shape})")
        else:
            nodecay_params.append(p)
            if master_process and print_names: print(f"NO DECAY: {pn} (shape: {p.shape})")

    optim_groups = [
        {'params': decay_params, 'weight_decay': wd_val},
        {'params': nodecay_params, 'weight_decay': 0.0} # No weight decay for these params
    ]

    if master_process:
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Optimizer: Decayed parameters: {num_decay_params:,} in {len(decay_params)} tensors.")
        print(f"Optimizer: Non-decayed parameters: {num_nodecay_params:,} in {len(nodecay_params)} tensors.")

    # Check if fused AdamW is available and use it if on CUDA for potential speedup.
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type_val == "cuda"
    if master_process: print(f"Optimizer: Using fused AdamW: {use_fused}")

    optimizer_instance = torch.optim.AdamW(optim_groups, lr=lr_val, betas=betas_val, eps=eps_val, fused=use_fused)
    return optimizer_instance

# Instantiate the optimizer using the raw model.
# `print_param_names` is True only for the master process to avoid verbose logging from all DDP ranks.
optimizer = configure_optimizers_refined(
    raw_model,
    weight_decay,
    max_lr,
    (adamw_beta1, adamw_beta2),
    adamw_eps,
    device_type,
    print_param_names=(master_process) # Set to True to see detailed param groups on master
)

# Learning rate scheduler function: Cosine decay with linear warmup.
def get_lr(it):
    """Calculates learning rate for a given step `it` based on warmup and cosine decay schedule."""
    # 1) Linear warmup for `warmup_steps`
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps  # (it+1) because step `it` starts from 0
    # 2) If `it` is beyond `max_steps`, return `min_lr` (constant after decay)
    if it > max_steps:
        return min_lr
    # 3) In between, use cosine decay down to `min_lr`
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1, f"Decay ratio out of bounds: {decay_ratio} for it={it}"
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Coefficient goes from 1 to 0
    return min_lr + coeff * (max_lr - min_lr)

if master_process:
    print("Optimizer and LR scheduler setup complete.")

# -----------------------------------------------------------------------------
# Training Loop

# Logging setup
log_dir = "log_olmo_custom"  # Directory to save logs and checkpoints
if master_process:
    os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist
    # Define log file path with a timestamp
    log_file_path = os.path.join(log_dir, f"training_log_{time.strftime('%Y%m%d-%H%M%S')}.txt")

    # Define header for the CSV log file.
    # `placeholder_custom_eval_metric_keys` should match keys returned by `evaluator.run_evaluations`.
    placeholder_custom_eval_metric_keys = [
        "hellaswag", "piqa", "openbookqa", "winogrande",
        "socialiqa", "commonsenseqa", "arc_easy", "arc_challenge"
    ] # Example metric keys, user should customize
    log_header_elements = [
        "step", "train_loss", "val_loss", "lr", "grad_norm",
        "dt_ms", "tok_per_sec"
    ] + placeholder_custom_eval_metric_keys

    # Write initial configuration and header to the log file
    with open(log_file_path, "w") as f:
        f.write(f"Starting training run with OLMo custom config.\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"OLMo Model Config: {str(olmo_model_config)}\n")
        f.write(f"Total Trainable Parameters: {count_parameters(raw_model):,}\n")
        f.write(f"Batch Size (Total Tokens): {total_batch_size}, Micro-Batch (B): {B}, Sequence Length (T): {T}, Grad Accum Steps: {grad_accum_steps}\n")
        f.write(f"Max Steps: {max_steps}, Warmup Steps: {warmup_steps}\n")
        f.write(f"Max LR: {max_lr:.2e}, Min LR: {min_lr:.2e}\n")
        f.write(f"Optimizer: AdamW, Weight Decay: {weight_decay}, Betas: ({adamw_beta1}, {adamw_beta2}), Eps: {adamw_eps}\n")
        f.write(f"Device: {device}, DDP World Size: {ddp_world_size}, Use Compile: {use_compile}\n")
        f.write("----------------------------------------------------------------------\n")
        f.write(",".join(log_header_elements) + "\n")  # CSV header line

if master_process:
    print(f"Starting training for {max_steps} steps...")
    print(f"Logging to: {log_file_path if 'log_file_path' in locals() else log_dir}")

# Ensure train_loader was initialized (it would have exited earlier if not)
# This is an extra check; primary check is during DataLoaderLite instantiation.
if train_loader is None and master_process: # Should not happen if previous checks worked
    print("Critical Error: Train loader is None at the start of training loop. Exiting.")
    if ddp: destroy_process_group()
    exit(1)

# Initialize val_loss_accum as a tensor on the correct device for DDP all_reduce compatibility.
# This variable is used for checkpointing if validation isn't run at step 0.
val_loss_for_checkpoint = torch.tensor(float('nan'), device=device)

# Main training loop iterates for `max_steps`
for step in range(max_steps):
    t0 = time.time()  # Start time for the current step
    last_step = (step == max_steps - 1)  # Flag for the very last step
    current_lr_for_log = get_lr(step)  # Get current LR for logging and optimizer update

    # Initialize dictionaries to store metrics for the current step's log entry
    current_eval_metrics = {key: "N/A" for key in placeholder_custom_eval_metric_keys}
    current_val_loss_str = "N/A" # Default validation loss string for logging

    # --- Evaluation Phase (Validation Loss & Custom Metrics) ---
    # This block runs periodically (e.g., every 250 steps) and on the last step.
    if step % 250 == 0 or last_step:
        if val_loader is not None:
            model.eval()  # Set model to evaluation mode (disables dropout, etc.)
            val_loader.reset()  # Reset validation data loader to its beginning
            with torch.no_grad():  # Disable gradient calculations for evaluation
                val_loss_accum_current_eval = torch.tensor(0.0, device=device) # Accumulator for current eval period's val loss
                val_loss_steps = 20  # Number of validation batches to average loss over
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    # Use mixed precision for evaluation if enabled for training
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y) # OLMo model returns (logits, loss)
                    loss = loss / val_loss_steps  # Normalize loss for averaging
                    val_loss_accum_current_eval += loss.detach() # Accumulate detached loss

            if ddp: # If DDP, average validation loss across all processes
                dist.all_reduce(val_loss_accum_current_eval, op=dist.ReduceOp.AVG)

            current_val_loss_str = f"{val_loss_accum_current_eval.item():.6f}"
            val_loss_for_checkpoint = val_loss_accum_current_eval.item() # Update for checkpointing

            if master_process:
                print(f"Step {step:5d} | Validation loss: {current_val_loss_str}")
        else:
            # val_loader is None (e.g., not provided or failed to initialize)
            if master_process: print(f"Step {step:5d} | Validation loader not available, skipping validation loss.")
            val_loss_for_checkpoint = float('nan') # Ensure it's float NaN for checkpoint

        # --- Custom Evaluation using `evaluator.py` ---
        if (evaluator is not None) and (not use_compile): # `use_compile` check from user's example
            if master_process: print(f"Running custom evaluations from 'evaluator.py' at step {step}...")

            # Call the user-defined `run_evaluations` function.
            # It's expected to return a dictionary of metrics for logging.
            returned_metrics = evaluator.run_evaluations(
                model=raw_model,             # Pass the unwrapped model
                config=olmo_model_config,    # Pass the model configuration
                device=device,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
                master_process=master_process, # `evaluator` should use this to control its own prints/writes
                log_file=log_file_path,      # Pass log file path for detailed logging by evaluator itself
                step=step
            )
            # Process returned metrics for CSV logging
            if master_process and returned_metrics and isinstance(returned_metrics, dict):
                print(f"Custom eval metrics reported: {returned_metrics}")
                for key in placeholder_custom_eval_metric_keys: # Iterate through expected keys for consistent CSV order
                    if key in returned_metrics:
                        metric_val = returned_metrics[key]
                        current_eval_metrics[key] = f"{metric_val:.4f}" if isinstance(metric_val, float) else str(metric_val)
                    # else: current_eval_metrics[key] remains "N/A"
            elif master_process:
                print("No metrics dictionary returned by evaluator, or evaluator returned None/empty.")
        elif master_process and evaluator is None:
            print("`evaluator.py` not found or not imported, skipping custom evaluations.")
        elif master_process and use_compile: # From user example, skip custom eval if compiled
            print("`use_compile` is True, skipping custom evaluations as per example logic.")

        # --- Model Checkpointing ---
        # Save a checkpoint periodically and on the last step (master process only).
        if step > 0 and (step % 5000 == 0 or last_step) and master_process: # Checkpointing frequency (e.g., every 5000 steps)
            checkpoint_path = os.path.join(log_dir, f"olmo_model_step{step:05d}.pt")
            checkpoint = {
                'model': raw_model.state_dict(),  # Model state dictionary
                'config': raw_model.config.to_dict() if hasattr(raw_model.config, 'to_dict') else raw_model.config, # Model config (serializable)
                'step': step,                     # Current training step
                'val_loss': val_loss_for_checkpoint, # Validation loss at this checkpoint
                'optimizer': optimizer.state_dict(),# Optimizer state for resuming training
                # Save key training parameters for easier resumption or analysis
                'total_batch_size': total_batch_size, 'B': B, 'T': T,
                'max_steps': max_steps, 'warmup_steps': warmup_steps, 'max_lr': max_lr, 'min_lr': min_lr,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # --- Training Phase (Forward Pass, Backward Pass, Optimizer Step) ---
    model.train()  # Set model to training mode (enables dropout, etc.)
    optimizer.zero_grad()  # Zero out gradients from the previous step before accumulation

    # Accumulator for training loss over gradient accumulation steps.
    # Initialize as a tensor on device for DDP all_reduce.
    loss_accum_train = torch.tensor(0.0, device=device)

    # Inner loop for gradient accumulation
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        is_last_micro_step = (micro_step == grad_accum_steps - 1)

        # DDP gradient synchronization control:
        # For nn.parallel.DistributedDataParallel, gradients are synced by default on each `backward()`.
        # To accumulate gradients locally before syncing, wrap forward and backward passes
        # in `model.no_sync()` context manager for all but the last micro-step.
        if ddp_world_size > 1 and not is_last_micro_step:
            with model.no_sync(): # Disable DDP gradient sync for this micro-step
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # Mixed precision
                    logits, loss = model(x, y)
                loss = loss / grad_accum_steps  # Normalize loss for accumulation
                loss_accum_train += loss.detach() # Accumulate loss (detached)
                loss.backward()  # Compute gradients (will be accumulated locally)
        else: # Not DDP, or DDP with world_size=1, or it's the last micro-step (time to sync)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # Mixed precision
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps  # Normalize loss
            loss_accum_train += loss.detach()
            loss.backward()  # Compute gradients (will sync if DDP and last micro-step)

    if ddp: # If DDP, average the accumulated training loss across all processes for consistent logging
        dist.all_reduce(loss_accum_train, op=dist.ReduceOp.AVG)

    # Gradient Clipping: Clip gradients to prevent exploding gradients.
    # `raw_model.parameters()` is used to ensure clipping is applied to the actual model parameters,
    # especially important when `model` is a DDP wrapper.
    norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0) # Clip norm set to 1.0

    # Update learning rate for the optimizer based on the current step and schedule.
    # `current_lr_for_log` was already calculated at the beginning of the step.
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr_for_log

    # Optimizer Step: Update model weights.
    optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete (good for accurate timing)

    t1 = time.time()  # End time for the current step
    dt_ms = (t1 - t0) * 1000  # Duration of the step in milliseconds

    # Calculate tokens per second for performance monitoring
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / (t1 - t0) if (t1 - t0) > 0 else 0

    # Logging training progress (master process only)
    if master_process:
        train_loss_str = f"{loss_accum_train.item():.6f}"
        grad_norm_str = f"{norm.item():.4f}" if norm is not None else "N/A" # norm can be tensor
        lr_str = f"{current_lr_for_log:.4e}"
        dt_ms_str = f"{dt_ms:.2f}"
        tok_per_sec_str = f"{tokens_per_sec:.0f}"

        # Prepare data for CSV logging, ensuring order matches `log_header_elements`
        log_data_elements = [
            str(step), train_loss_str, current_val_loss_str, lr_str,
            grad_norm_str, dt_ms_str, tok_per_sec_str
        ]
        for key in placeholder_custom_eval_metric_keys: # Append custom metrics in defined order
            log_data_elements.append(str(current_eval_metrics.get(key, "N/A")))

        # Write log entry to file
        with open(log_file_path, "a") as f:
            f.write(",".join(log_data_elements) + "\n")

        # Print concise log to console
        print(f"Step {step:5d} | Train Loss: {train_loss_str} | Val Loss: {current_val_loss_str} | LR: {lr_str} | Grad Norm: {grad_norm_str} | dt: {dt_ms_str}ms | Tokens/sec: {tok_per_sec_str}")

# --- Sample Generation (at the end of training) ---
# This block runs only on the master process, if not using `torch.compile`, and if `enc` (tokenizer) is available.
if master_process and (not use_compile) and enc:
    model.eval()  # Set model to evaluation mode
    print("\n--- Generating Samples from Trained Model ---")
    num_return_sequences = 4  # Number of independent samples to generate
    max_length_generation = 32 # Max length of each generated sample (including prompt) - keep short for quick check

    # Example prompt
    prompt_text = "Hello, I'm a language model,"
    prompt_tokens = enc.encode(prompt_text)
    prompt_tokens_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    # Repeat prompt for batch generation if desired, here generating one by one for simplicity in loop
    # prompt_tokens_tensor = prompt_tokens_tensor.unsqueeze(0).repeat(num_return_sequences, 1)

    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank) # ddp_rank is 0 here (master_process)

    generated_sequences_text = []
    for i in range(num_return_sequences):
        current_sequence = prompt_tokens_tensor.unsqueeze(0) # Start with the prompt (B=1, T_prompt)

        # Generate tokens one by one up to `max_length_generation`
        for _ in range(max_length_generation - len(prompt_tokens)): # Generate remaining tokens
            if current_sequence.size(1) >= olmo_model_config.max_sequence_length: # Safety break for model context limit
                break

            with torch.no_grad(): # No gradients needed for generation
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # Mixed precision
                    # OLMo model expects `input_ids`
                    outputs = model(input_ids=current_sequence)
                    # `outputs` from OLMoForCausalLM is a CausalLMOutputWithPast object,
                    # which contains `logits`.
                    logits = outputs.logits

                next_token_logits = logits[:, -1, :]  # Get logits for the very last token position
                probs = F.softmax(next_token_logits, dim=-1) # Convert logits to probabilities

                # Sample from top-k probabilities for diversity
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # Top-k sampling (k=50)
                next_token_id_sampled = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng) # Sample one token
                actual_next_token_id = torch.gather(topk_indices, -1, next_token_id_sampled) # Get the actual token ID from top-k indices

                # Append the sampled token to the current sequence
                current_sequence = torch.cat((current_sequence, actual_next_token_id), dim=1)

        # Decode the generated token sequence to text
        generated_tokens_list = current_sequence[0, :max_length_generation].tolist()
        decoded_text = enc.decode(generated_tokens_list)
        print(f"Sample {i+1}: {decoded_text}")
        generated_sequences_text.append(decoded_text)
    print("--------------------------------------------")

# --- End of Training ---
if master_process:
    print("Training complete.")
    if 'log_file_path' in locals(): print(f"Log file saved to: {log_file_path}")

    # Save a final model checkpoint
    final_checkpoint_path = os.path.join(log_dir, "olmo_model_final.pt")
    final_checkpoint = {
        'model': raw_model.state_dict(),
        'config': raw_model.config.to_dict() if hasattr(raw_model.config, 'to_dict') else raw_model.config,
        'step': step, # Final step number
        'val_loss': val_loss_for_checkpoint if 'val_loss_for_checkpoint' in locals() and isinstance(val_loss_for_checkpoint, float) else float('nan'),
        'optimizer': optimizer.state_dict(),
        'total_batch_size': total_batch_size, 'B': B, 'T': T,
        'max_steps': max_steps, 'warmup_steps': warmup_steps, 'max_lr': max_lr, 'min_lr': min_lr,
    }
    torch.save(final_checkpoint, final_checkpoint_path)
    print(f"Final checkpoint saved to {final_checkpoint_path}")

# Clean up DDP resources if DDP was enabled
if ddp:
    destroy_process_group()

# End of script
