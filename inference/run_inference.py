#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)


def build_prefix(prompt: str) -> str:
    # Keep inference prompt contract aligned with training.
    return f"Prompt:\n{prompt}\n\nGenerate structured SVG targets.\n\n"


def read_rows(csv_path: Path, id_col: str, prompt_col: str) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get(id_col, "")
            prompt = row.get(prompt_col, "")
            if prompt:
                rows.append({"sample_id": sample_id, "prompt": prompt})
    return rows


def maybe_init_distributed(torch_mod):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not torch_mod.distributed.is_initialized():
        backend = "nccl" if torch_mod.cuda.is_available() else "gloo"
        torch_mod.distributed.init_process_group(backend=backend)
    return rank, local_rank, world_size


def pick_dtype(dtype_name: str, torch_mod):
    name = dtype_name.lower()
    if name == "bf16":
        return torch_mod.bfloat16
    if name == "fp16":
        return torch_mod.float16
    if name == "fp32":
        return torch_mod.float32
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run quick sanity inference and save raw generations."
    )
    parser.add_argument(
        "--model-path",
        default="../models/best_model",
        help="Path to local fine-tuned model directory",
    )
    parser.add_argument(
        "--prompts-csv",
        default="../training/final_training.csv",
        help="CSV containing prompt/id columns",
    )
    parser.add_argument("--id-col", default="id", help="ID column name in CSV")
    parser.add_argument("--prompt-col", default="prompt", help="Prompt column name in CSV")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="How many prompts to run (recommended 20-50)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument(
        "--output-csv",
        default="outputs/raw_generations.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep intermediate per-rank shard CSV files",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max generated tokens per sample",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (ignored if --do-sample is not set)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p (ignored if --do-sample is not set)",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable stochastic sampling (default is greedy decoding)",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Torch dtype override for model weights",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, local_rank, world_size = maybe_init_distributed(torch)

    model_path = Path(args.model_path).resolve()
    prompts_csv = Path(args.prompts_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")
    if not prompts_csv.exists():
        raise SystemExit(f"Prompts CSV does not exist: {prompts_csv}")

    rows = read_rows(prompts_csv, args.id_col, args.prompt_col)
    if not rows:
        raise SystemExit("No prompt rows found. Check --prompts-csv and --prompt-col.")

    rng = random.Random(args.seed)
    sample_size = max(1, min(args.sample_size, len(rows)))
    sampled = rng.sample(rows, sample_size)
    local_sampled = sampled[rank::world_size]

    if rank == 0:
        print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = pick_dtype(args.dtype, torch)
    load_kwargs = {"trust_remote_code": True}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        load_kwargs["torch_dtype"] = torch.bfloat16

    if rank == 0:
        print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if torch.cuda.is_available():
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    print(
        f"[rank {rank}/{world_size}] device={device} "
        f"global_samples={len(sampled)} local_samples={len(local_sampled)}"
    )

    out_rows: list[dict] = []
    with torch.no_grad():
        for i, row in enumerate(local_sampled, 1):
            prompt = row["prompt"]
            prefix = build_prefix(prompt)
            enc = tokenizer(prefix, return_tensors="pt").to(device)

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": args.do_sample,
            }
            if args.do_sample:
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = args.top_p

            output_ids = model.generate(**enc, **gen_kwargs)
            continuation_ids = output_ids[0][enc["input_ids"].shape[1] :]
            generated_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)

            out_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "prompt": prompt,
                    "generated_text": generated_text,
                }
            )
            print(f"[rank {rank}] [{i}/{len(local_sampled)}] sample_id={row['sample_id']}")

    shard_csv = output_csv.with_suffix(f".rank{rank}.csv")
    with shard_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "prompt", "generated_text"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[rank {rank}] Saved shard: {shard_csv}")

    if world_size > 1:
        torch.distributed.barrier()

    if rank == 0:
        merged_rows: list[dict] = []
        for r in range(world_size):
            shard = output_csv.with_suffix(f".rank{r}.csv")
            if not shard.exists():
                continue
            with shard.open("r", newline="", encoding="utf-8") as f:
                merged_rows.extend(csv.DictReader(f))

        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "prompt", "generated_text"])
            writer.writeheader()
            writer.writerows(merged_rows)

        if world_size > 1 and not args.keep_shards:
            for r in range(world_size):
                shard = output_csv.with_suffix(f".rank{r}.csv")
                if shard.exists():
                    shard.unlink()

        print(f"Saved {len(merged_rows)} generations to: {output_csv}")

    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
