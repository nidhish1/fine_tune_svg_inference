# FineTune_SVG_NG

A fine-tuned text-to-SVG model for generating icon-style SVG from natural language prompts.

## Overview

This model is a continuation fine-tune focused on improving SVG serialization completion.

Pipeline:

- Prompt -> model generation (`structured` mode)
- Extract SVG block
- Validate XML + rule checks
- Lightweight repair fallback for truncated outputs
- Final `id,svg` submission file

## Training Summary

- Base approach: Causal LM fine-tuning for text-to-SVG
- Target focus: `serialization_only` continuation stage
- Data scale: ~50k train rows, ~5k validation rows
- Hardware: multi-GPU training (`bf16`)
- Objective emphasis: next-token LM loss; auxiliary heads reduced/disabled in late continuation runs to focus on SVG completion

## Inference Setup

Typical decode settings:

- `max_new_tokens=1536`
- `dtype=bf16`
- `output_mode=structured`
- deterministic decoding (`do_sample=False`) for stable outputs

Multi-GPU inference is supported via `torchrun`.

## Post-processing / Validation

Generated outputs are post-processed to recover valid SVG when generations are truncated:

- detect SVG region
- trim trailing broken tail
- close obvious dangling tags
- enforce XML parseability

Primary checks:

- valid XML with `<svg>` root
- max length constraints
- max `<path>` count
- allowed SVG tags

## Expected Files

Model folder (example):

- `best_model/config.json`
- `best_model/model.safetensors`
- `best_model/tokenizer.json`
- `best_model/tokenizer_config.json`
- `best_model/generation_config.json`

Input CSV:

- columns: `sample_id,prompt`

Output CSV:

- raw: `sample_id,prompt,generated_text`
- final submission: `id,svg`

## Quick Inference Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference/run_inference.py \
  --model-path /path/to/best_model \
  --prompts-csv /path/to/test_processed.csv \
  --id-col sample_id \
  --prompt-col prompt \
  --sample-size 1000 \
  --max-new-tokens 1536 \
  --dtype bf16 \
  --output-mode structured \
  --output-csv /path/to/raw_generations_test_1000.csv
```

## Notes

- Raw generations can include incomplete/truncated SVG; use validation+repair before final submission.
- For competition upload, ensure final file is exactly:
  - columns: `id,svg`
  - row coverage: matches test set IDs exactly once.
