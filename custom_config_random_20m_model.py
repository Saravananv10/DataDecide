# 1. Import necessary libraries
from hf_olmo.modeling_olmo import OLMoConfig, OLMoForCausalLM
import torch # Needed for model instantiation and parameter counting

# 2. Explicitly define the model configuration
# These values are chosen to be similar to a ~20M parameter model.
# You can modify these to change the model architecture and total parameters.
# Refer to the OLMoConfig documentation or source for all available options.
# Based on allenai/DataDecide-dolma1_7-20M config.json:
# "d_model": 192, "n_heads": 8, "n_layers": 16, "mlp_ratio": 8, "vocab_size": 50304 (matches GPT-NeoX)
# "embedding_size": 50304 (often same as vocab_size)

# Core architectural parameters
vocab_size = 50304  # Size of the vocabulary (e.g., GPT-NeoX tokenizer)
embedding_size = vocab_size # Usually same as vocab_size, or d_model if distinct projection
d_model = 192       # Dimension of the model (hidden size)
n_layers = 16       # Number of transformer layers
n_heads = 8         # Number of attention heads
mlp_ratio = 8       # Ratio for the feed-forward network hidden size (d_ff = d_model * mlp_ratio)
# d_ff = d_model * mlp_ratio # Can also be set directly via mlp_hidden_size

# Other important parameters (can be adjusted)
max_sequence_length = 1024 # Max input sequence length
weight_tying = True       # Whether to tie input and output embeddings (saves parameters)
attention_dropout = 0.0   # Dropout rate for attention probabilities
residual_dropout = 0.0    # Dropout rate for residual connections
embedding_dropout = 0.0   # Dropout rate for embeddings
ff_dropout = 0.0          # Dropout for feed-forward layers (was called mlp_dropout in some configs)
                               # OLMoConfig uses 'residual_dropout' for most, and specific ones for att/ff. Check its definition.
                               # For OLMoConfig, it seems 'dropout' is a general one, then also 'attention_dropout', 'multi_query_attention_dropout'

# Let's align with OLMoConfig fields more directly:
# We will use some defaults from OLMoConfig and override specific ones.
# Default OLMoConfig might have d_model=768, n_layers=12 etc.
# We are aiming for a smaller model.

print("Defining custom OLMo configuration...")
custom_config = OLMoConfig(
    # Architecture
    d_model=d_model,
    n_layers=n_layers,
    n_heads=n_heads,
    mlp_ratio=mlp_ratio, # or set mlp_hidden_size = d_model * mlp_ratio
    # Vocabulary and Embeddings
    vocab_size=vocab_size,
    embedding_size=embedding_size, # If None, defaults to d_model. For explicit vocab projection, set to vocab_size.
                                   # If weight_tying=True, embedding_size must match d_model if no separate projection.
                                   # Let's ensure this. If embedding_size is vocab_size and distinct from d_model,
                                   # it implies input/output projection layers.
                                   # For simplicity and parameter saving with tying, often embedding_size = d_model
    weight_tying=weight_tying,     # Tie input and output token embeddings
    # Sequence length
    max_sequence_length=max_sequence_length,
    # Regularization (set to 0.0 for no dropout initially if desired)
    attention_dropout=attention_dropout,  # Corresponds to `attn_pdrop` in some other models
    residual_dropout=residual_dropout,    # Corresponds to `resid_pdrop`
    embedding_dropout=embedding_dropout,  # Corresponds to `embd_pdrop`
    # ff_dropout seems to be covered by residual_dropout or a general 'dropout' in OLMo.
    # Let's check OLMoConfig source for precise dropout fields:
    # It has: dropout, attention_dropout, multi_query_attention_dropout
    # We'll set the general 'dropout' which might apply to ff and other places.
    dropout=0.0, # General dropout rate, potentially for FF layers if not specified otherwise

    # Other OLMo specific fields if needed (using defaults for now)
    # e.g., flash_attention, multi_query_attention, rotary_percentage, rope_theta etc.
    # Set flash_attention=False if not available or for wider compatibility initially
    flash_attention=False, # Requires specific CUDA setup
    # multi_query_attention=False, # Set to True if you want MQA
    # clip_qkv=None,
    # attention_layer_norm=False, # If True, adds LayerNorm to QKV projections
)

# If weight_tying is True, OLMo expects embedding_size to be d_model.
# If you want a distinct projection for vocab and then tie, the logic is more complex.
# For standard weight tying where token embeddings are directly used as output layer:
if custom_config.weight_tying:
    custom_config.embedding_size = custom_config.d_model

print("Custom configuration created:")
# print(custom_config) # You can print the whole config

# 3. Instantiate the model with the custom configuration
print("\nInitializing model with random weights using the custom configuration...")
try:
    model = OLMoForCausalLM(custom_config)
    model.train() # Set model to training mode (affects dropout, batchnorm etc. if any)
except Exception as e:
    print(f"Error initializing model from custom configuration: {e}")
    exit()
print("Model initialized successfully.")

# 4. Count and print the total number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"\nTotal trainable parameters: {total_params:,}")

# 5. Verify model initialization (optional)
# Print out some parameter values to see they are random.
try:
    example_weights = model.transformer.blocks[0].att_proj.weight.data
    print("\nExample weights from the first layer's attention projection (should be random):")
    print(example_weights[:2, :5])

    # Final output layer
    # If weight_tying=True, the output weights are tied to the input token embeddings.
    if custom_config.weight_tying:
        output_layer_weights = model.get_input_embeddings().weight.data
        print("\nExample weights from the (tied) output embedding layer (should be random):")
    else:
        # OLMoForCausalLM uses self.transformer.fc_out as the final projection if not tied
        output_layer_weights = model.transformer.fc_out.weight.data
        print("\nExample weights from the final output projection layer (fc_out, should be random):")
    print(output_layer_weights[:2, :5])

except AttributeError as e:
    print(f"\nCould not access example weights for verification: {e}")
    print("This might be due to model structure details. The model is still initialized randomly.")
except Exception as e:
    print(f"\nError during weight verification: {e}")


print("\n\nTo run this script:")
print("1. Save it as a Python file (e.g., custom_config_random_20m_model.py).")
print("2. Install the necessary libraries: pip install ai2-olmo torch")
print("3. Modify the hyperparameters in the 'Explicitly define the model configuration' section as needed.")
print("4. Run the script: python custom_config_random_20m_model.py")
print("   The script will print the total number of parameters for your configuration.")
print("\nNext steps would involve preparing your dataset and writing a training loop for this custom model.")
