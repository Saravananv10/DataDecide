# 1. Import necessary libraries
from hf_olmo.modeling_olmo import OLMoConfig # Using OLMoConfig directly
from hf_olmo import OLMoForCausalLM
from transformers import AutoConfig # Still useful for potentially fetching config dict

# 2. Specify the model name for configuration reference
# We'll use this to fetch the configuration details consistent with the 20M models.
# You can choose any of the 20M models from the DataDecide suite for their config,
# e.g., "allenai/DataDecide-dolma1_7-20M".
config_reference_model_name = "allenai/DataDecide-dolma1_7-20M"
# As identified before, "step14584-seed0" is a valid revision for 20M models.
# The configuration should be consistent across revisions for a given model size.
revision_name = "step14584-seed0"

print(f"Loading configuration for model: {config_reference_model_name}, revision: {revision_name}")

try:
    # Load the configuration object from the pretrained model
    # This ensures our new model has the same architecture (hidden size, num layers, etc.)
    config = AutoConfig.from_pretrained(config_reference_model_name, revision=revision_name)

    # Ensure it's an OLMoConfig or convert if necessary.
    # The AutoConfig might return a generic Config dictionary,
    # but OLMoForCausalLM expects an OLMoConfig instance.
    if not isinstance(config, OLMoConfig):
        # If it's a dictionary-like object from AutoConfig, pass its attributes to OLMoConfig
        # You might need to explicitly map attributes if names differ or add missing ones
        # For OLMo, critical attributes are often like d_model, n_layers, n_heads, etc.
        # Let's check the OLMoConfig definition or the config.json of the reference model
        # to ensure all required fields are present.

        # A safer way for OLMo is to load its specific config if possible,
        # or ensure all necessary fields from its config.json are used.
        # The `hf_olmo.OLMoConfig.from_pretrained` method is what we should use.
        olmo_config = OLMoConfig.from_pretrained(config_reference_model_name, revision=revision_name)

    else:
        olmo_config = config

    print("Configuration loaded successfully:")
    # print(olmo_config) # You can print the whole config to inspect

except ImportError:
    print("Error: ai2-olmo or transformers library not found.")
    print("Please install them by running:")
    print("pip install ai2-olmo transformers")
    exit()
except Exception as e:
    print(f"Error loading configuration: {e}")
    print(f"Make sure the model name '{config_reference_model_name}' and revision '{revision_name}' are correct.")
    exit()

print("\nInitializing model with random weights using the loaded configuration...")

try:
    # Instantiate the model using the loaded configuration.
    # This will initialize the model with random weights.
    model = OLMoForCausalLM(olmo_config)
except Exception as e:
    print(f"Error initializing model from configuration: {e}")
    exit()

print("Model initialized with random weights successfully.")

# 4. Verify model initialization (optional)
# Print out some parameter values to see they are random.
# For example, the weights of the first layer's query projection.
# The exact name of parameters can vary, inspect model.state_dict().keys() if unsure.
try:
    # Example: accessing parameters of the first transformer block's attention mechanism
    # The exact path might differ based on the OLMo model structure.
    # You might need to inspect `model.named_parameters()` to find a suitable tensor.
    example_weights = model.transformer.blocks[0].att_proj.weight.data
    print("\nExample weights from the first layer (should be random):")
    print(example_weights[:2, :5]) # Print a small slice

    # Another check: model's output embedding layer
    if hasattr(model, 'embed_out') and model.embed_out is not None: # For OLMo, it's often model.transformer.ff_out
      # The final layer is often tied to input embeddings or is a separate Linear layer.
      # In OLMo, the final projection might be part of the last transformer block or a separate layer.
      # Let's look at the output projection `model.transformer.ff_out` if it exists, or `embed_out`
      # The model card for OLMo shows it's a standard decoder-only transformer.
      # The final linear layer projecting to vocabulary size is usually `lm_head` or similar.
      # In OLMo, it's `model.transformer.fc_out` based on common patterns, or it might be tied.
      # According to OLMo source, `OLMoForCausalLM` adds a `classifier` head (self.transformer.fc_out)
      # if `config.embedding_size != config.d_model` and `config.weight_tying` is False.
      # Otherwise, if weight_tying is true, it uses the input embeddings.
      # Let's assume `model.transformer.fc_out` or a similar named final projection.
      # If `config.weight_tying` is true, `model.get_output_embeddings().weight` would be same as input.
      # If not, there's a separate output layer.
      # The `hf_olmo` `OLMoForCausalLM` class has `self.transformer.fc_out` as the final linear layer to vocab size.
      if hasattr(model.transformer, 'fc_out') and model.transformer.fc_out is not None:
          final_layer_weights = model.transformer.fc_out.weight.data
          print("\nExample weights from the final output projection layer (should be random):")
          print(final_layer_weights[:2, :5])
      elif olmo_config.weight_tying and hasattr(model.transformer, 'wte'): # Tied weights
          final_layer_weights = model.transformer.wte.weight.data
          print("\nWeights from the (tied) output embedding layer (same as input, should be random if new):")
          print(final_layer_weights[:2, :5])
      else:
          print("\nCould not find a distinct final output projection layer for quick inspection (might be tied or named differently).")


except AttributeError as e:
    print(f"\nCould not access example weights for verification: {e}")
    print("This might be due to model structure variations. The model is still initialized randomly.")
except Exception as e:
    print(f"\nError during weight verification: {e}")


print("\n\nTo run this script:")
print("1. Save it as a Python file (e.g., initialize_random_20m_model.py).")
print("2. Install the necessary libraries: pip install ai2-olmo transformers torch")
print("3. Run the script: python initialize_random_20m_model.py")
print("\nNext steps would involve preparing your dataset and writing a training loop.")
