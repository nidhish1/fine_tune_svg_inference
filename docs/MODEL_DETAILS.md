# Model Details

## Model Name

**FineTune_SVG_NG_Qwen2.5-3B**

## Subtitle

Serialization-Focused Qwen2.5-3B Fine-Tune for Reliable Text-to-SVG Icon Generation

## Checkpoint Source

Local model used for upload:

- `models/serialModel/best_model`

## Overview

This model is fine-tuned for text-to-SVG generation, with emphasis on improving serialization completion and producing structurally valid SVG outputs.

## Intended Use

- Input: natural-language prompt describing an icon/scene
- Output: SVG-oriented text generation
- Best use: structured generation + validation/repair before submission

## Training Summary (High Level)

- Architecture family: Qwen2.5-3B causal LM
- Fine-tuning focus: serialization-oriented continuation
- Precision: `bf16`
- Multi-GPU training and inference workflow

## Inference Settings (Recommended)

- `output_mode=structured`
- `max_new_tokens=1536`
- `dtype=bf16`
- deterministic decoding (`do_sample=False`) for stable outputs

## I/O Format

### Input CSV

- `sample_id,prompt`

### Raw Output CSV

- `sample_id,prompt,generated_text`

### Final Submission CSV

- `id,svg`

## Validation Constraints (Competition-Oriented)

- valid XML with `<svg>` root
- max SVG length (validator limit)
- max `<path>` count
- allowed SVG tag list

## Known Limitations

- Raw generations may include truncation artifacts
- Some rows can need post-processing repair
- Hybrid fallback strategy may be needed for highest submission robustness

## Notes

For best leaderboard stability, run generation first, then apply strict XML/constraint validation and lightweight repair before exporting final `id,svg`.
