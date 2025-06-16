# 1. Import necessary libraries
from hf_olmo import OLMoForCausalLM
from transformers import AutoTokenizer

# 2. Specify the model name
# You can find other 20M models in the model card:
# https://huggingface.co/allenai/DataDecide-dolma1_7-20M (check the table)
model_name = "allenai/DataDecide-dolma1_7-20M"
# From the model card, the 20M models were trained for 1.9B tokens.
# The checkpoint naming convention seems to be step<num_tokens_trained_in_billions*10000>-...
# For 1.9B tokens, this would be around step19000.
# Looking at the available revisions/branches on Hugging Face for this model,
# e.g., https://huggingface.co/allenai/DataDecide-dolma1_7-20M/branches,
# a relevant revision is "step14584-seed0" (1.9B tokens / ~130 tokens per step batch = ~14615 steps)
# The model card hyperparameters table also shows "Training steps: 14,584" for 20M models.
# We'll use a specific revision to ensure reproducibility.
# Let's pick the default seed one.
revision_name = "step14584-seed0" # Or another specific step/seed if preferred

print(f"Loading model: {model_name}, revision: {revision_name}")

# 3. Load the pre-trained model and tokenizer
try:
    # It's good practice to specify cache_dir if you want to control where models are downloaded.
    # model = OLMoForCausalLM.from_pretrained(model_name, revision=revision_name, cache_dir="./hf_cache")
    # tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision_name, cache_dir="./hf_cache")
    # For simplicity in this example, let's not specify cache_dir and use the default.
    model = OLMoForCausalLM.from_pretrained(model_name, revision=revision_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision_name)
except ImportError:
    print("Error: ai2-olmo or transformers library not found.")
    print("Please install them by running:")
    print("pip install ai2-olmo transformers")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you have a working internet connection and the model name/revision is correct.")
    print(f"You can check available revisions at https://huggingface.co/{model_name}/tree/main or /branches")
    exit()

print("Model and tokenizer loaded successfully.")

# 4. Create a sample prompt
prompt = "The Allen Institute for AI (AI2) is known for"

# 5. Generate text
# Encode the prompt
inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)

print(f"Generating text for prompt: '{prompt}'")

# Generate text
# You can adjust max_length, temperature, top_k, top_p, etc.
# For a small model like 20M, keeping max_length reasonable is good.
try:
    response = model.generate(**inputs, max_length=50, temperature=0.7, do_sample=True)
    # Decode the generated ids to text
    generated_text = tokenizer.decode(response[0], skip_special_tokens=True)
except Exception as e:
    print(f"Error during text generation: {e}")
    exit()

# 6. Print the generated text
print("\nGenerated Text:")
print(generated_text)

print("\n\nTo run this script:")
print("1. Save it as a Python file (e.g., run_20m_model.py).")
print("2. Install the necessary libraries: pip install ai2-olmo transformers torch")
print("3. Run the script: python run_20m_model.py")
