# Inference Next Steps

## 1) Pull and freeze artifacts locally

- Copy `best_model` and `best_metrics.json` into `models/`.
- Also copy `train.log` and chart PNGs for experiment tracking.

## 2) Run a quick inference sanity pass (20-50 prompts)

- Generate outputs from `models/best_model`.
- Save raw generations with at least: `sample_id`, `prompt`, `generated_text`.

Example command (from `inference/`):

```bash
python3 run_inference.py \
  --model-path ../models/best_model \
  --prompts-csv ../training/final_training.csv \
  --sample-size 30 \
  --output-csv outputs/raw_generations.csv
```

## 3) Post-process to final SVG-only output

- If output includes structured blocks, extract content after `serialization_target:`.
- Enforce strict `<svg ...>...</svg>` extraction.
- Trim trailing junk outside the closing `</svg>`.

## 4) Run validity checks on generated SVGs

- XML parse check.
- Rule checks:
  - root tag must be `svg`
  - max char limit
  - max path limit
  - allowed tag list
- Track summary metrics:
  - validity rate
  - parse fail rate
  - average char length
  - average path count

## 5) Run visual/render spot-check

- Render 50-100 samples to PNG (or browser preview).
- Inspect for:
  - blank outputs
  - repeated templates
  - malformed geometry

## 6) Check fallback risk explicitly

- Hash generated SVGs and flag high duplicate rate.
- Count exact matches to known fallback/template strings.
- Check diversity across prompts.

## 7) Pick final checkpoint formally

- Current winner: around step `2400` (`checkpoint_score=0.9070`).
- If needed, compare against one earlier checkpoint on a fixed eval prompt set.

## 8) Prepare submission package

- Include inference script and model path config.
- Ensure required output format contract is met.
- Add a small README with exact run command, dependencies, and runtime notes.

## 9) Create a reproducibility snapshot

- Save:
  - full training config
  - random seed
  - hardware info
  - commit/version info
- Store as `run_summary.json` next to model artifacts.

## 10) Optional final decoding calibration

- Tune decoding settings (for example: temperature, top-p, max_new_tokens, repetition_penalty).
- Pick settings balancing:
  - validity
  - diversity
  - prompt faithfulness

