import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
# from hellaswag import render_example, iterate_examples # Evaluation part, to be commented out

# OLMo imports
from olmo.config import ModelConfig, OptimizerConfig, SchedulerConfig, DataConfig, TrainConfig
from olmo.model import OLMo
from olmo.optim import build_optimizer
from olmo.tokenizer import Tokenizer

# Placeholder for DDP
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
# ModelConfig (formerly GPTConfig)
# All parameters from the original GPTConfig will be mapped here,
# or OLMo defaults will be used.
# Original GPTConfig fields were:
# block_size: int = 1024 # max sequence length
# vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
# n_layer: int = 12
# n_head: int = 12
# n_embd: int = 768
# ---- OLMo ModelConfig equivalent ----
# d_model: int # n_embd
# n_layers: int # n_layer
# n_heads: int # n_head
# vocab_size: int
# max_sequence_length: int # block_size
# Other OLMo specific params:
# embedding_size: Optional[int] = 50257 (will use vocab_size)
# activation_type: ActivationType = "swiglu"
# block_type: BlockType = "sequential"
# rope: bool = True
# flash_attention: bool = False (keep default)
# attention_dropout: float = 0.1 (keep default)
# residual_dropout: float = 0.1 (keep default)
# embedding_dropout: float = 0.1 (keep default)
# layer_norm_type: str = "default" (keep default)
# include_bias: bool = True (keep default)

# Note: The original GPTConfig is removed as per instructions.
# We will directly instantiate ModelConfig.

# -----------------------------------------------------------------------------
# OLMo Model (formerly GPT model)
# The CausalSelfAttention, MLP, Block, and GPT classes are removed.
# OLMo class from olmo.model will be used.

# -----------------------------------------------------------------------------
# Optimizer configuration

def configure_optimizers(model: OLMo, olmo_train_config: TrainConfig, device_type: str):
    """
    Configures the optimizer for the OLMo model.
    This function creates an AdamW optimizer based on the OLMo model's parameters
    and optimizer settings from an OLMo-style TrainConfig.
    It refers to olmo.optim.build_optimizer for parameter grouping logic.
    """
    # The build_optimizer function from olmo.optim will handle parameter grouping (decay vs. no_decay)
    # and other optimizer specific configurations based on TrainConfig.

    # Log the usage of fused AdamW if applicable (build_optimizer handles this internally if 'fused':True in args)
    if device_type == 'cuda':
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        if fused_available and olmo_train_config.optimizer.args.get("fused", False):
             print("Requesting fused AdamW via TrainConfig.")
        elif not fused_available and olmo_train_config.optimizer.args.get("fused", False):
            print("Fused AdamW requested but not available in this PyTorch version.")
        else:
            print("Fused AdamW not requested or not applicable.")

    optimizer = build_optimizer(config=olmo_train_config, model=model)

    # Print summary of parameters from build_optimizer logic (if possible, or replicate for info)
    # This part is tricky as build_optimizer doesn't return this info directly.
    # For now, we'll rely on build_optimizer's internal logic.
    # We can print general info though.
    print(f"Optimizer configured: {olmo_train_config.optimizer.name}")
    print(f"Learning rate: {olmo_train_config.optimizer.learning_rate}, Weight decay: {olmo_train_config.optimizer.weight_decay}")

    return optimizer

# -----------------------------------------------------------------------------
# Main script part (adapted from a typical GPT-2 training script structure)

if __name__ == "__main__":
    # Minimal execution test parameters
    run_minimal_execution_test = True # Set to False to run the placeholder loop

if run_minimal_execution_test:
    print("RUNNING MINIMAL EXECUTION TEST")
    device = "cpu"
    # DDP is off for this test
    ddp = False
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0
    ddp_local_rank = 0

    # Tiny model configuration for the test
    init_from = "scratch" # Force scratch for tiny model
    tiny_model_config_args = dict(
        d_model=64,
        n_layers=2,
        n_heads=2,
        vocab_size=50304, # Must match tokenizer
        embedding_size=50304,
        max_sequence_length=64, # For dummy data
        activation_type="swiglu",
        block_type="sequential",
        rope=True,
        eos_token_id=OLMO_EOS_TOKEN_ID, # Defined globally
        pad_token_id=OLMO_PAD_TOKEN_ID, # Defined globally
    )
    # Override general model/data params for the tiny test
    model_constructor_args = tiny_model_config_args
    data_loader_sequence_length = 32 # Must be <= tiny_model_config_args.max_sequence_length
    batch_size = 2 # Small batch for testing

    # Optimizer params (can keep as is or simplify further if needed for CPU)
    learning_rate = 3e-4
    weight_decay = 0.1
    beta1 = 0.9
    beta2 = 0.95
else:
    # DDP setup (placeholder) - original logic
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # init_process_group(backend='nccl')
        # ddp_rank = int(os.environ['RANK'])
        # ddp_local_rank = int(os.environ['LOCAL_RANK'])
        # ddp_world_size = int(os.environ['WORLD_SIZE'])
        # device = f'cuda:{ddp_local_rank}'
        # torch.cuda.set_device(device)
        # master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        print("DDP setup placeholder - full script mode")
        master_process = True # for non-ddp run
        seed_offset = 0
        # ddp_world_size = 1
    else:
        master_process = True
        seed_offset = 0
        # ddp_world_size = 1

    # Default parameters (can be overridden by command line args if this were a full script)
    # General training parameters
    batch_size = 16 # Original default
    data_loader_sequence_length = 1024 # Original default
    # vocab_size = 50304 # Defined in ModelConfig
    # --- OLMo ModelConfig parameters ---
    d_model = 768 # Original default
    n_layers = 12 # Original default
    n_heads = 12  # Original default

    # Optimizer parameters (for the simple AdamW) ---
    learning_rate = 3e-4 # from original prompt
    weight_decay = 0.1   # from original prompt
    beta1 = 0.9
    beta2 = 0.95

    # device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    # dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

    # Init OLMo ModelConfig
    # These will be used to create the ModelConfig instance
    model_constructor_args = dict(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=50304, # Must match tokenizer
        embedding_size=50304,
        max_sequence_length=data_loader_sequence_length, # Use data_loader_sequence_length here
        activation_type="swiglu",
        block_type="sequential",
        rope=True,
        eos_token_id=OLMO_EOS_TOKEN_ID, # Defined globally
        pad_token_id=OLMO_PAD_TOKEN_ID, # Defined globally
    )
    init_from = 'scratch' # or 'olmo-1b', 'olmo-7b' (will use placeholder from_pretrained)


# --- Common setup for both minimal test and full script placeholder ---

# Tokenizer specific IDs (using OLMo defaults) are defined globally now
# OLMO_EOS_TOKEN_ID = 50256
# OLMO_PAD_TOKEN_ID = 50256

# Initialize the OLMo tokenizer
# Replacing tiktoken.get_encoding("gpt2")
print(f"Initializing OLMo Tokenizer (allenai/olmo-1b)...")
# Note: This requires internet access the first time to download tokenizer files.
# And the `tokenizers` library installed (dependency of ai2-olmo).
enc = Tokenizer.from_pretrained(
    "allenai/olmo-1b", # Using 1B for tokenizer is fine for the test too
    eos_token_id=OLMO_EOS_TOKEN_ID,
    pad_token_id=OLMO_PAD_TOKEN_ID
)
print(f"OLMo Tokenizer initialized. Vocab size: {enc.vocab_size}, EOS: {enc.eos_token_id}, PAD: {enc.pad_token_id}")


# Initialize model from scratch or load from pretrained
if init_from == 'scratch':
    # For the minimal test, model_constructor_args is already tiny_model_config_args
    print(f"Initializing a new OLMo model from scratch with args: {model_constructor_args}")
    olmo_model_config = ModelConfig(**model_constructor_args)
    model = OLMo(olmo_model_config)
elif init_from.startswith('olmo-'): # This branch not used in minimal test if init_from is forced to "scratch"
    print(f"Initializing OLMo model from pretrained: {init_from}")
    model = OLMo.from_pretrained(model_type=init_from)
    olmo_model_config = model.config
    if olmo_model_config.eos_token_id != OLMO_EOS_TOKEN_ID or olmo_model_config.pad_token_id != OLMO_PAD_TOKEN_ID:
        print(f"Warning: Pretrained model's token IDs differ from globally set IDs.")
else:
    raise ValueError(f"Unknown init_from: {init_from}")

# Ensure DataLoader sequence length is compatible with model's max_sequence_length
# Note: data_loader_sequence_length is set based on run_minimal_execution_test
assert data_loader_sequence_length <= olmo_model_config.max_sequence_length, \
    f"Data sequence length (T={data_loader_sequence_length}) " \
    f"must be <= OLMo model max_sequence_length ({olmo_model_config.max_sequence_length})"
if master_process:
    print(f"Verified: Data sequence length ({data_loader_sequence_length}) <= OLMo model max_sequence_length ({olmo_model_config.max_sequence_length})")

model.to(device if run_minimal_execution_test else 'cuda') # Simplified device handling for test

if master_process:
    print("OLMo model architecture:")
    # print(model) # Can be very verbose for larger models
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# Optimizer
opt_config = OptimizerConfig(
    name="AdamW",
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    betas=(beta1, beta2),
    args={"fused": True if not run_minimal_execution_test else False} # Fused only if not CPU test
)
dummy_scheduler_config = SchedulerConfig(name="constant_with_warmup", t_warmup="0ba", t_max="1ba")
dummy_data_config = DataConfig(paths=[])

olmo_train_config = TrainConfig(
    model=olmo_model_config,
    optimizer=opt_config,
    scheduler=dummy_scheduler_config,
    data=dummy_data_config,
    device_train_batch_size=batch_size,
    device_train_grad_accum=1,
)

# device_type for fused AdamW check in configure_optimizers
current_device_type = device if run_minimal_execution_test else ('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = configure_optimizers(model, olmo_train_config, current_device_type)

raw_model = model # No DDP wrapping in minimal test

# --- Minimal Execution Test Logic ---
if run_minimal_execution_test and master_process:
    print("\n--- Starting Minimal Execution Test ---")

    # 4. Forward/Backward Pass Test
    print("\n--- Testing Forward/Backward Pass ---")
    B, T = batch_size, data_loader_sequence_length # Use tiny test settings
    dummy_x = torch.randint(0, olmo_model_config.vocab_size, (B, T), device=device)
    dummy_y = torch.randint(0, olmo_model_config.vocab_size, (B, T), device=device)
    model.train() # Ensure model is in training mode

    try:
        # Forward pass
        output = model(input_ids=dummy_x)
        logits = output.logits
        print(f"Logits shape: {logits.shape}") # Expected: (B, T, vocab_size)

        # Compute loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), dummy_y.view(-1))
        print(f"Loss: {loss.item()}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        print("Backward pass successful.")

        # Optimizer step
        optimizer.step()
        print("Optimizer step successful.")
        print("--- Forward/Backward Pass Test Successful ---")
    except Exception as e:
        print(f"!!! Forward/Backward Pass Test Failed: {e} !!!")
        # raise e # Optionally re-raise

    # 5. Generation Test
    print("\n--- Testing Generation ---")
    model.eval() # Ensure model is in eval mode for generation
    prompt_text = "Hello, world"
    max_gen_tokens = 5
    temperature_gen = 0.8

    try:
        # The generate function needs the tokenizer `enc` and `device`
        # Let's update its signature or pass them appropriately.
        # For now, assume `generate` can access `enc` and `device` if they are global,
        # or pass them. The current `generate` func in script doesn't take them.
        # We'll modify `generate` to accept tokenizer and device.

        print(f"Prompt: '{prompt_text}'")
        generated_ids = generate(
            model_instance=model,
            tokenizer=enc,
            prompt=prompt_text,
            max_new_tokens=max_gen_tokens,
            temperature=temperature_gen,
            device=device
        )
        generated_text = enc.decode(generated_ids[0].tolist()) # Assuming generate returns a batch
        print(f"Generated text: '{generated_text}'")
        print("--- Generation Test Successful (ran without error) ---")
    except Exception as e:
        print(f"!!! Generation Test Failed: {e} !!!")
        # raise e # Optionally re-raise

    print("\n--- Minimal Execution Test Finished ---")

# --- Original Placeholder Training Loop (conditional) ---
elif master_process: # Only if not run_minimal_execution_test
    print("\nStarting placeholder training loop (SKIPPED for minimal test run)...")
    # The original loop was here. For clarity, it's removed if minimal test is active.
    # If you want to keep it, wrap it with `if not run_minimal_execution_test:`
    # for iter_num in range(max_iters):
    # ... (rest of the original loop) ...
    # if master_process:
    #    print("Placeholder training loop finished.")


# -----------------------------------------------------------------------------
# Evaluation related functions (commented out as per instructions)
# This section demonstrates how generation would adapt.
# The actual `generate` function in a full script would be more complex.

@torch.no_grad()
def generate(model_instance: OLMo, tokenizer: Tokenizer, prompt: str, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None, device: str = "cpu"):
    """
    Generates text from a prompt using the OLMo model and tokenizer.
    """
    model_instance.eval() # Ensure model is in eval mode
    # Encode the prompt string to token IDs
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) # OLMo tokenizer might not need add_special_tokens control like this
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0) # Add batch dimension

    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= model_instance.config.max_sequence_length else idx[:, -model_instance.config.max_sequence_length:]
        # forward the model to get the logits for the index in the sequence
        output = model_instance(input_ids=idx_cond) # OLMo.forward takes input_ids
        logits = output.logits
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    model_instance.train() # Return model to train mode if it was in it before
    return idx # Return IDs including prompt

# def get_most_likely_row(tokens, mask, logits):
# ... (rest of the script, DDP finalization, print "Script finished", etc.) ...
if run_minimal_execution_test:
    print("Minimal test script part finished.")
else:
    # DDP finalization for the full script mode
    if ddp:
        # destroy_process_group()
        pass
    print("Full script placeholder finished.")

    # Future steps from the prompt to integrate:
    # ...
```
