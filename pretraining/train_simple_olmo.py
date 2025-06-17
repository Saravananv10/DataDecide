import argparse
import os
import time
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from datasets import load_dataset
from hf_olmo import OLMoConfig, OLMoForCausalLM

# Helper function to count trainable parameters

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Predefined model configs matching DataDecide hyperparameters
OLMO_MODEL_CONFIGURATIONS = {
    "4M": {"d_model": 64, "n_layers": 8, "n_heads": 8},
    "6M": {"d_model": 96, "n_layers": 8, "n_heads": 8},
    "8M": {"d_model": 128, "n_layers": 8, "n_heads": 8},
    "10M": {"d_model": 144, "n_layers": 8, "n_heads": 8},
    "14M": {"d_model": 192, "n_layers": 8, "n_heads": 8},
    "16M": {"d_model": 208, "n_layers": 8, "n_heads": 8},
    "20M": {"d_model": 192, "n_layers": 16, "n_heads": 8},
    "60M": {"d_model": 384, "n_layers": 16, "n_heads": 12},
    "90M": {"d_model": 528, "n_layers": 16, "n_heads": 12},
    "150M": {"d_model": 768, "n_layers": 12, "n_heads": 12},
    "300M": {"d_model": 1024, "n_layers": 16, "n_heads": 16},
    "530M": {"d_model": 1344, "n_layers": 16, "n_heads": 16},
    "750M": {"d_model": 1536, "n_layers": 16, "n_heads": 16},
    "1B": {"d_model": 2048, "n_layers": 16, "n_heads": 16},
}

OLMO_1B_MLP_RATIO = 2.75  # Align with released OLMo-1B config


# Tokenize and group text into fixed-length blocks

def tokenize_function(example, tokenizer):
    return tokenizer(example["text"])


def group_texts(examples, block_size):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [
            t[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def prepare_dataset(dataset_name, tokenizer, block_size):
    ds = load_dataset(dataset_name, split="train")
    ds = ds.map(lambda ex: tokenize_function(ex, tokenizer), batched=True, remove_columns=["text"])
    ds = ds.map(lambda ex: group_texts(ex, block_size), batched=True)
    ds.set_format(type="torch")
    return ds


def create_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def main():
    parser = argparse.ArgumentParser(description="Train a small OLMo model")
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--dataset", type=str, help="HF dataset name or local path")
    parser.add_argument("--model_size", type=str, default="20M", choices=list(OLMO_MODEL_CONFIGURATIONS.keys()))
    parser.add_argument("--output_dir", type=str, default="./olmo_ckpt")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8.4e-3)
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--eval_tasks", nargs="*", help="List of lm_eval task names")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k) and v is not None:
                setattr(args, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_params = OLMO_MODEL_CONFIGURATIONS[args.model_size]

    tokenizer_name = "allenai/OLMo-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size
    embedding_size = config_params["d_model"]

    mlp_ratio = OLMO_1B_MLP_RATIO if args.model_size == "1B" else 8.0

    config = OLMoConfig(
        d_model=config_params["d_model"],
        n_layers=config_params["n_layers"],
        n_heads=config_params["n_heads"],
        mlp_ratio=mlp_ratio,
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        max_sequence_length=args.block_size,
        weight_tying=True,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        dropout=0.0,
        flash_attention=False,
        init_device="cpu",
    )
    config.precision = "fp32"
    config.torch_dtype = torch.float32

    print("Initializing model...")
    model = OLMoForCausalLM(config).to(device)
    print(f"Model has {count_parameters(model):,} trainable parameters")

    dataset = prepare_dataset(args.dataset, tokenizer, args.block_size)
    dataloader = create_dataloader(dataset, args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    global_step = 0
    model.train()

    t0 = time.time()
    while global_step < args.max_steps:
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()

            if (global_step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            t1 = time.time()
            dt = t1 - t0
            tokens_processed = args.batch_size * args.block_size * args.grad_accum_steps
            tokens_per_sec = tokens_processed / dt
            print(
                f"step {global_step:5d} | loss: {loss.item():.6f} | lr {optimizer.param_groups[0]['lr']:.4e} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )
            t0 = time.time()

            if global_step >= args.max_steps:
                break

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")

    if args.eval_tasks:
        try:
            from lm_eval import evaluator

            results = evaluator.simple_evaluate(
                model=model,
                tokenizer=tokenizer,
                tasks=args.eval_tasks,
                batch_size=args.batch_size,
                num_fewshot=0,
            )
            out_file = os.path.join(args.output_dir, "eval_results.yaml")
            with open(out_file, "w") as f:
                yaml.safe_dump(results, f)
            print(f"Evaluation results written to {out_file}")
        except Exception as e:
            print(f"Evaluation skipped due to error: {e}")


if __name__ == "__main__":
    main()
